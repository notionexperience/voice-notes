// Cloudflare Worker — receives a voice recording and files it in a Notion database.
// Audio arrives as the raw request body; metadata travels in headers.
//
// Pipeline: auth → transcribe (Whisper) → upload audio + summarise + punctuate
//           (all in parallel) → create the row.
// Every AI step is free, key-less and non-blocking: if one fails the note is
// still created with its audio attached.

const NOTION_API = "https://api.notion.com/v1";
const DEFAULT_LLM = "@cf/meta/llama-3.1-8b-instruct-fp8-fast";

const EXT_BY_MIME = {
  "audio/webm": "weba",
  "audio/ogg": "ogg",
  "audio/mp4": "m4a",
  "audio/mpeg": "mp3",
  "audio/wav": "wav",
  "audio/x-wav": "wav",
  "audio/aac": "aac",
  "audio/flac": "flac",
};

const corsHeaders = (env) => ({
  "Access-Control-Allow-Origin": env.ALLOWED_ORIGIN || "*",
  "Access-Control-Allow-Headers": "content-type, x-app-secret, x-title, x-tags",
  "Access-Control-Allow-Methods": "POST, OPTIONS",
});

const json = (env, status, body) =>
  new Response(JSON.stringify(body), {
    status,
    headers: { "content-type": "application/json", ...corsHeaders(env) },
  });

async function notion(env, path, { method = "GET", body, form } = {}) {
  const headers = {
    Authorization: `Bearer ${env.NOTION_TOKEN}`,
    "Notion-Version": env.NOTION_VERSION || "2026-03-11",
  };
  if (body) headers["Content-Type"] = "application/json";
  const res = await fetch(`${NOTION_API}${path}`, {
    method,
    headers,
    body: form || (body ? JSON.stringify(body) : undefined),
  });
  const data = await res.json().catch(() => ({}));
  if (!res.ok) throw new Error(`Notion ${path} ${res.status}: ${data.message || "request failed"}`);
  return data;
}

async function resolveDataSourceId(env) {
  if (env.NOTION_DATA_SOURCE_ID) return env.NOTION_DATA_SOURCE_ID;
  const dbId = (env.NOTION_DATABASE_ID || "").replace(/-/g, "");
  if (!dbId) throw new Error("Missing NOTION_DATABASE_ID");
  const db = await notion(env, `/databases/${dbId}`);
  const id = db.data_sources?.[0]?.id;
  if (!id) throw new Error("Database has no data source");
  return id;
}

function pickProp(schema, type, override) {
  if (override && schema[override]?.type === type) return override;
  return Object.keys(schema).find((k) => schema[k].type === type) || null;
}

// Match a property by its exact name — use this when a database has several
// properties of the same type (e.g. Pin / Archive / Audio are all checkboxes).
function byName(schema, name, type) {
  return name && schema[name]?.type === type ? name : null;
}

// ---- Text and block helpers --------------------------------------------

const chunk = (text, size = 1800) => text.match(new RegExp(`[\\s\\S]{1,${size}}`, "g")) || [];
const words = (s) => s.split(/\s+/).filter(Boolean).length;
const rich = (content) => [{ type: "text", text: { content } }];

const heading = (content) => ({
  object: "block",
  type: "heading_3",
  heading_3: { rich_text: rich(content) },
});
const paragraph = (content) => ({
  object: "block",
  type: "paragraph",
  paragraph: { rich_text: rich(content) },
});
const bullet = (content) => ({
  object: "block",
  type: "bulleted_list_item",
  bulleted_list_item: { rich_text: rich(content.slice(0, 1800)) },
});
const todo = (content) => ({
  object: "block",
  type: "to_do",
  to_do: { rich_text: rich(content.slice(0, 1800)), checked: false },
});

// Respect the paragraph breaks the clean-up step introduced, but never emit
// more blocks than Notion will accept in one request.
const transcriptBlocks = (text, limit = 95) => {
  const blocks = text
    .split(/\n{2,}/)
    .map((p) => p.trim())
    .filter(Boolean)
    .flatMap((p) => chunk(p).map(paragraph));
  if (blocks.length <= limit) return blocks;
  // Too many short paragraphs: re-slice evenly so a long note still fits.
  const size = Math.min(1800, Math.max(200, Math.ceil(text.length / limit)));
  return chunk(text, size).slice(0, limit).map(paragraph);
};

const decodeHeader = (v) => {
  try {
    return decodeURIComponent(v || "").trim();
  } catch {
    return "";
  }
};

// Words Whisper reliably mis-hears. Priming the model with them, and naming
// them again during clean-up, fixes "motion" / "ocean" for "Notion".
// Override with the VOCABULARY variable: a comma-separated list.
const vocab = (env) =>
  (env.VOCABULARY || "Notion, Notion AI, Obsidian, Apple Notes")
    .split(",")
    .map((t) => t.trim())
    .filter(Boolean);

// ---- Transcription (free, Cloudflare Workers AI) ------------------------

// btoa() chokes on very large argument lists, so encode in chunks.
function toBase64(buffer) {
  const bytes = new Uint8Array(buffer);
  const CHUNK = 0x8000;
  let binary = "";
  for (let i = 0; i < bytes.length; i += CHUNK) {
    binary += String.fromCharCode.apply(null, bytes.subarray(i, i + CHUNK));
  }
  return btoa(binary);
}

async function transcribe(env, blob) {
  if (!env.AI) return "";
  // Encoding burns CPU and the Workers free plan allows 10 ms per request,
  // so skip transcription for long recordings. The audio still uploads.
  const maxBytes = Number(env.TRANSCRIBE_MAX_MB || 3) * 1024 * 1024;
  if (blob.size > maxBytes) return "";
  try {
    const model = env.CF_TRANSCRIBE_MODEL || "@cf/openai/whisper-large-v3-turbo";
    const buffer = await blob.arrayBuffer();
    // whisper-large-v3-turbo takes base64; the older @cf/openai/whisper takes bytes.
    const input = model.includes("turbo")
      ? { audio: toBase64(buffer), task: "transcribe", vad_filter: true }
      : { audio: [...new Uint8Array(buffer)] };
    if (env.TRANSCRIBE_LANGUAGE) input.language = env.TRANSCRIBE_LANGUAGE;
    const terms = vocab(env);
    // Supported by whisper-large-v3-turbo; biases decoding toward these words.
    if (terms.length && model.includes("turbo")) {
      input.initial_prompt = `Terms used in this recording: ${terms.join(", ")}.`;
    }
    const out = await env.AI.run(model, input);
    return (out?.text || "").trim();
  } catch {
    return "";
  }
}

// The recorder normally slices long audio and base64-encodes it itself.
// That keeps this Worker's CPU cost near zero, which is what removes the size
// ceiling: each slice is its own request with its own CPU budget.
// Errors are thrown, not swallowed, so the recorder knows to fall back.
async function transcribeBase64(env, base64) {
  if (!env.AI) throw new Error("Workers AI binding is not configured");
  const model = env.CF_TRANSCRIBE_MODEL || "@cf/openai/whisper-large-v3-turbo";
  if (!model.includes("turbo")) throw new Error("Chunked transcription needs a turbo Whisper model");
  const input = { audio: base64, task: "transcribe", vad_filter: true };
  if (env.TRANSCRIBE_LANGUAGE) input.language = env.TRANSCRIBE_LANGUAGE;
  const terms = vocab(env);
  if (terms.length) input.initial_prompt = `Terms used in this recording: ${terms.join(", ")}.`;
  const out = await env.AI.run(model, input);
  return (out?.text || "").trim();
}

// ---- Transcript clean-up (free, Cloudflare Workers AI) ------------------
// Whisper returns one long unpunctuated lowercase run. This restores sentences
// and paragraphs without rewording. Chunks run in parallel, so the whole step
// costs roughly the latency of a single call.
// Set POLISH_ENABLED="false" to keep the raw Whisper output.

function polishPrompt(env) {
  const terms = vocab(env);
  return [
    "You restore punctuation and capitalisation in speech-to-text output.",
    "Rules:",
    "- Keep every word the speaker said. Never summarise, reorder, add or delete content.",
    "- Add full stops, commas, question marks and capital letters.",
    "- Start a new paragraph when the topic shifts. Separate paragraphs with a blank line.",
    "- Remove filler only when it is pure noise: um, uh, er.",
    terms.length
      ? `- The speaker uses these terms: ${terms.join(", ")}. Speech-to-text often mis-hears them as similar-sounding words (for example "motion" or "ocean" instead of "Notion"). Restore the correct term wherever the context makes it clear.`
      : "- Capitalise obvious product and brand names.",
    "Return the corrected text only. No preamble, no quotes, no commentary.",
  ].join("\n");
}

async function polishChunk(env, model, text) {
  const out = await env.AI.run(model, {
    messages: [
      { role: "system", content: polishPrompt(env) },
      { role: "user", content: text },
    ],
    max_tokens: Math.min(4000, Math.ceil(text.length / 2) + 200),
    temperature: 0.1,
  });
  const cleaned = (typeof out?.response === "string" ? out.response : "").trim();
  // A model that truncates or editorialises is worse than no clean-up at all,
  // so only accept output that still has roughly the original word count.
  const ratio = cleaned ? words(cleaned) / Math.max(1, words(text)) : 0;
  return ratio >= 0.75 && ratio <= 1.35 ? cleaned : text;
}

async function polish(env, transcript) {
  if (!env.AI || !transcript) return transcript;
  if (String(env.POLISH_ENABLED).toLowerCase() === "false") return transcript;
  if (transcript.length > Number(env.POLISH_MAX_CHARS || 40000)) return transcript;
  const model = env.SUMMARY_MODEL || DEFAULT_LLM;
  try {
    const parts = chunk(transcript, 2000);
    // Bound the fan-out. Anything past the cap keeps its raw wording.
    const head = parts.slice(0, 24);
    const tail = parts.slice(24).join("");
    const cleaned = await Promise.all(head.map((p) => polishChunk(env, model, p)));
    return cleaned.join("\n\n") + (tail ? "\n\n" + tail : "");
  } catch {
    return transcript;
  }
}

// ---- Summarisation (free, Cloudflare Workers AI) ------------------------
// Produces a title, a short summary, key points and action items.
// Set SUMMARY_ENABLED="false" to switch this off.

const SUMMARY_SCHEMA = {
  type: "object",
  properties: {
    title: { type: "string" },
    summary: { type: "string" },
    key_points: { type: "array", items: { type: "string" } },
    actions: { type: "array", items: { type: "string" } },
  },
  required: ["title", "summary", "key_points", "actions"],
};

function summaryPrompt(env) {
  const language = env.SUMMARY_LANGUAGE
    ? `Write every field in ${env.SUMMARY_LANGUAGE}.`
    : "Write in the same language the speaker used.";
  return [
    "You turn rambling voice notes into structured notes. The transcript is speech-to-text and may contain mis-hearings.",
    "Return JSON only, with these keys:",
    '- "title": a specific headline of at most 8 words. No quotes, no trailing period.',
    '- "summary": 2-4 sentences covering what was actually said, including the specifics. Name the items rather than counting them.',
    '- "key_points": an array of the note\'s substance: facts, observations, opinions, decisions, details worth keeping.',
    '- "actions": an array of short imperative tasks. Include anything the speaker wants to make, do, buy, fix, follow up on or decide. When the note is a list of ideas or plans, each entry becomes an action.',
    "Rules:",
    "- Never place the same item in both key_points and actions. Choose the better fit.",
    "- Never invent detail. Keep names, numbers and specifics exactly as spoken.",
    "- Use [] for an empty array, never null.",
    `- Spell these terms exactly this way: ${vocab(env).join(", ")}.`,
    language,
  ].join("\n");
}

// Loose comparison so an item never appears as both a bullet and a checkbox.
const normalise = (s) =>
  s
    .toLowerCase()
    .replace(/[^a-z0-9 ]/g, "")
    .replace(/\s+/g, " ")
    .trim();

function parseSummary(out) {
  let data = out?.response ?? out;
  if (typeof data === "string") {
    const match = data.match(/\{[\s\S]*\}/);
    if (!match) return null;
    try {
      data = JSON.parse(match[0]);
    } catch {
      return null;
    }
  }
  if (!data || typeof data !== "object") return null;

  const clean = (v) => (typeof v === "string" ? v.trim() : "");
  const list = (v) => (Array.isArray(v) ? v.map(clean).filter(Boolean) : []);

  const actions = list(data.actions).slice(0, 25);
  const seen = actions.map(normalise);
  const keyPoints = list(data.key_points)
    .filter((p) => {
      const n = normalise(p);
      return !seen.some((a) => a === n || a.includes(n) || n.includes(a));
    })
    .slice(0, 25);

  const result = {
    title: clean(data.title).replace(/^["']|["'.]+$/g, "").slice(0, 120),
    summary: clean(data.summary),
    keyPoints,
    actions,
  };
  return result.summary || result.keyPoints.length || result.actions.length ? result : null;
}

async function summarize(env, transcript) {
  if (!env.AI || !transcript) return null;
  if (String(env.SUMMARY_ENABLED).toLowerCase() === "false") return null;
  // A one-line note is already its own summary.
  if (words(transcript) < Number(env.SUMMARY_MIN_WORDS || 25)) return null;

  const model = env.SUMMARY_MODEL || DEFAULT_LLM;
  const messages = [
    { role: "system", content: summaryPrompt(env) },
    { role: "user", content: transcript.slice(0, Number(env.SUMMARY_MAX_CHARS || 12000)) },
  ];
  const base = { messages, max_tokens: 900, temperature: 0.2 };

  try {
    const out = await env.AI.run(model, {
      ...base,
      response_format: { type: "json_schema", json_schema: SUMMARY_SCHEMA },
    });
    const parsed = parseSummary(out);
    if (parsed) return parsed;
  } catch {
    // Not every model honours response_format — fall through to a plain call.
  }

  try {
    return parseSummary(await env.AI.run(model, base));
  } catch {
    return null;
  }
}

// ---- Notion upload -----------------------------------------------------

async function uploadAudio(env, file, filename, mime) {
  const upload = await notion(env, "/file_uploads", {
    method: "POST",
    body: { filename, content_type: mime },
  });
  const form = new FormData();
  form.append("file", file, filename);
  await notion(env, `/file_uploads/${upload.id}/send`, { method: "POST", form });
  return upload;
}

async function handleUpload(request, env) {
  if (!env.NOTION_TOKEN) throw new Error("Missing NOTION_TOKEN");
  if (env.APP_SECRET && request.headers.get("x-app-secret") !== env.APP_SECRET) {
    return json(env, 401, { ok: false, error: "Unauthorized" });
  }

  // Two accepted shapes. Multipart is what the recorder sends today and can
  // carry a transcript it produced itself; a raw audio body keeps older
  // recorders working after a Worker-only deploy.
  const contentType = request.headers.get("content-type") || "";
  let audio;
  let mime;
  let typedTitle = decodeHeader(request.headers.get("x-title"));
  let rawTags = decodeHeader(request.headers.get("x-tags"));
  let clientTranscript = "";

  if (contentType.includes("multipart/form-data")) {
    const form = await request.formData();
    audio = form.get("audio");
    if (!audio || typeof audio === "string") throw new Error("Missing audio file");
    mime = (audio.type || "audio/webm").split(";")[0];
    typedTitle = String(form.get("title") || typedTitle || "").trim();
    rawTags = String(form.get("tags") || rawTags || "");
    clientTranscript = String(form.get("transcript") || "").trim();
  } else {
    mime = contentType.split(";")[0] || "audio/webm";
    audio = await request.blob();
  }
  if (!audio.size) throw new Error("Empty recording");

  const ext = EXT_BY_MIME[mime] || "weba";
  const stamp = new Date().toISOString().replace(/[:.]/g, "-").slice(0, 19);
  const filename = `voice-note-${stamp}.${ext}`;
  const file = new File([audio], filename, { type: mime });

  const [dataSourceId, serverTranscript] = await Promise.all([
    resolveDataSourceId(env),
    // Only transcribe here when the recorder could not do it itself.
    clientTranscript ? Promise.resolve("") : transcribe(env, file),
  ]);
  const raw = clientTranscript || serverTranscript;

  // Schema read, byte push and clean-up are independent.
  const [ds, upload, transcript] = await Promise.all([
    notion(env, `/data_sources/${dataSourceId}`),
    uploadAudio(env, file, filename, mime),
    polish(env, raw),
  ]);

  // Deliberately serial: summarising the corrected text stops Whisper's
  // mis-hearings leaking into the summary and the action items.
  const summary = await summarize(env, transcript || raw);

  const schema = ds.properties || {};
  const titleProp = pickProp(schema, "title", env.NOTION_TITLE_PROP);
  const filesProp = pickProp(schema, "files", env.NOTION_AUDIO_PROP);
  const dateProp = pickProp(schema, "date", env.NOTION_DATE_PROP);
  const tagsProp = pickProp(schema, "multi_select", env.NOTION_TAGS_PROP);
  const audioFlagProp = byName(schema, env.NOTION_AUDIO_FLAG_PROP || "Audio", "checkbox");
  const summaryProp = byName(schema, env.NOTION_SUMMARY_PROP || "Summary", "rich_text");

  // Title: what the user typed → what the model named it → first words → timestamp.
  const title =
    typedTitle ||
    summary?.title ||
    (transcript ? transcript.slice(0, 80).replace(/\s+\S*$/, "") : "") ||
    `Voice note — ${new Date().toLocaleString("en-GB", { timeZone: env.TIMEZONE || "UTC" })}`;

  const properties = {};
  if (titleProp) properties[titleProp] = { title: [{ text: { content: title.slice(0, 200) } }] };
  if (dateProp) properties[dateProp] = { date: { start: new Date().toISOString() } };
  if (filesProp) {
    properties[filesProp] = {
      files: [{ type: "file_upload", file_upload: { id: upload.id }, name: filename }],
    };
  }
  const tags = rawTags
    .split(",")
    .map((t) => t.trim())
    .filter(Boolean);
  if (tagsProp && tags.length) properties[tagsProp] = { multi_select: tags.map((name) => ({ name })) };
  if (audioFlagProp) properties[audioFlagProp] = { checkbox: true };
  if (summaryProp && summary?.summary) {
    properties[summaryProp] = { rich_text: rich(summary.summary.slice(0, 2000)) };
  }

  // Page body: audio → summary → key points → action items → transcript.
  const children = [
    { object: "block", type: "audio", audio: { type: "file_upload", file_upload: { id: upload.id } } },
  ];

  if (summary?.summary) {
    children.push(heading("Summary"));
    for (const part of chunk(summary.summary)) children.push(paragraph(part));
  }
  if (summary?.keyPoints.length) {
    children.push(heading("Key points"));
    for (const point of summary.keyPoints) children.push(bullet(point));
  }
  if (summary?.actions.length) {
    children.push(heading("Action items"));
    for (const item of summary.actions) children.push(todo(item));
  }
  if (transcript) {
    // Notion accepts 100 blocks per request, nested children included.
    const body = transcriptBlocks(transcript, Math.max(1, 96 - children.length));
    if (summary) {
      // Tuck the raw text away so the summary is what you read first.
      children.push({
        object: "block",
        type: "toggle",
        toggle: { rich_text: rich("Transcript"), children: body },
      });
    } else {
      children.push(heading("Transcript"), ...body);
    }
  }

  const page = await notion(env, "/pages", {
    method: "POST",
    body: {
      parent: { type: "data_source_id", data_source_id: dataSourceId },
      icon: { type: "emoji", emoji: "🎙️" },
      properties,
      children: children.slice(0, 100), // Notion accepts 100 blocks per request
    },
  });

  return json(env, 200, {
    ok: true,
    url: page.url,
    title,
    transcribed: Boolean(raw),
    summarized: Boolean(summary),
    polished: Boolean(raw) && transcript !== raw,
    chunked: Boolean(clientTranscript),
  });
}

export default {
  async fetch(request, env) {
    const url = new URL(request.url);

    // One audio slice in, one piece of text out. The recorder calls this
    // repeatedly so recording length stops being limited by Worker CPU time.
    if (url.pathname === "/api/transcribe") {
      if (request.method === "OPTIONS") {
        return new Response(null, { status: 204, headers: corsHeaders(env) });
      }
      if (request.method !== "POST") return json(env, 405, { ok: false, error: "Use POST" });
      if (env.APP_SECRET && request.headers.get("x-app-secret") !== env.APP_SECRET) {
        return json(env, 401, { ok: false, error: "Unauthorized" });
      }
      try {
        const base64 = (await request.text()).trim();
        if (!base64) throw new Error("Empty chunk");
        return json(env, 200, { ok: true, text: await transcribeBase64(env, base64) });
      } catch (err) {
        return json(env, 500, { ok: false, error: err.message });
      }
    }

    if (url.pathname === "/api/send-to-notion") {
      if (request.method === "OPTIONS") {
        return new Response(null, { status: 204, headers: corsHeaders(env) });
      }
      if (request.method !== "POST") return json(env, 405, { ok: false, error: "Use POST" });
      try {
        return await handleUpload(request, env);
      } catch (err) {
        return json(env, 500, { ok: false, error: err.message });
      }
    }

    return env.ASSETS.fetch(request);
  },
};
