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
  "Access-Control-Allow-Methods": "GET, POST, OPTIONS",
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

// ---- Install self-check -------------------------------------------------
// A read-only diagnosis of a fresh install. It never writes to Notion, so it
// is safe to run as often as you like. public/check.html renders the result as
// a green / red checklist, which answers most setup questions without support.

const MIB = 1048576;
// Two real recordings from this app measured 129 and 215 kbps. Used only to
// turn a byte limit into "about N minutes".
const BYTES_PER_MIN_LOW = 967500;
const BYTES_PER_MIN_HIGH = 1612500;
// Notion accepts a one-shot upload of at most 20 MiB, whatever the plan allows.
const SINGLE_PART_MAX = 20 * MIB;

const minutesFor = (bytes) =>
  `${Math.floor(bytes / BYTES_PER_MIN_HIGH)}\u2013${Math.floor(bytes / BYTES_PER_MIN_LOW)} minutes`;

// Same call as notion(), but a failure comes back as data instead of throwing —
// a check that fails still has to report the ones after it.
async function notionProbe(env, path) {
  try {
    return { ok: true, data: await notion(env, path) };
  } catch (err) {
    const code = /\s(\d{3}):\s/.exec(err.message);
    return { ok: false, status: code ? Number(code[1]) : 0, message: err.message };
  }
}

async function runChecks(env) {
  const checks = [];
  const add = (id, label, status, detail, fix = "", optional = false) =>
    checks.push({ id, label, status, detail, fix, optional });

  // --- 1. Answerable without leaving the Worker --------------------------

  if (env.APP_SECRET) {
    add("app_secret", "Passphrase is set", "pass", "Requests without the passphrase are rejected.");
  } else {
    add(
      "app_secret",
      "Passphrase is set",
      "fail",
      "APP_SECRET is empty, so anyone who finds this web address can write into your Notion database.",
      "Cloudflare dashboard \u2192 Workers & Pages \u2192 your Worker \u2192 Settings \u2192 Variables and Secrets \u2192 Add. Name it APP_SECRET, choose Secret, paste a long random passphrase, then Deploy.",
    );
  }

  add(
    "ai_binding",
    "Workers AI is connected",
    env.AI ? "pass" : "fail",
    env.AI
      ? "Transcription, clean-up and summaries are available."
      : "No AI binding found. Notes would still save, but with audio only \u2014 no transcript and no summary.",
    env.AI ? "" : 'wrangler.jsonc must contain "ai": { "binding": "AI" }. Put it back and deploy again.',
  );

  try {
    const now = new Date().toLocaleString("en-GB", { timeZone: env.TIMEZONE || "UTC" });
    add(
      "timezone",
      "Time zone",
      env.TIMEZONE ? "pass" : "warn",
      env.TIMEZONE
        ? `TIMEZONE is ${env.TIMEZONE}, so an auto-named note reads ${now}.`
        : `TIMEZONE is not set, so auto-named notes use UTC and read ${now}.`,
      env.TIMEZONE ? "" : "Set the TIMEZONE variable to your IANA zone, e.g. Europe/Berlin or America/New_York.",
      !env.TIMEZONE,
    );
  } catch (err) {
    add(
      "timezone",
      "Time zone",
      "fail",
      `\u201c${env.TIMEZONE}\u201d is not a valid IANA time zone, so dates cannot be formatted.`,
      "Use a Region/City name with the same capitals, e.g. Europe/Berlin, America/New_York, Asia/Tokyo.",
    );
  }

  // --- 2. The token ------------------------------------------------------

  const token = (env.NOTION_TOKEN || "").trim();
  const tokenFix =
    "In Notion: Settings \u2192 Connections \u2192 Develop or manage integrations \u2192 New integration \u2192 pick your workspace \u2192 Save, then copy the Internal Integration Secret. Paste it into Cloudflare \u2192 your Worker \u2192 Settings \u2192 Variables and Secrets \u2192 NOTION_TOKEN, and Deploy.";

  if (!token) {
    add("token", "Notion token is present", "fail", "NOTION_TOKEN is empty.", tokenFix);
  } else if (!/^(ntn_|secret_)/.test(token)) {
    add(
      "token",
      "Notion token is present",
      "warn",
      "NOTION_TOKEN does not start with ntn_ or secret_, so it may be the wrong value.",
      tokenFix,
      true,
    );
  } else {
    add("token", "Notion token is present", "pass", "Looks like an internal integration secret.");
  }

  let integrationName = "your integration";
  let workspaceName = "";
  let tokenWorks = false;
  let maxUpload = 0;

  if (token) {
    const me = await notionProbe(env, "/users/me");
    if (me.ok) {
      tokenWorks = true;
      integrationName = me.data.name || integrationName;
      workspaceName = me.data.bot?.workspace_name || "";
      maxUpload = Number(me.data.bot?.workspace_limits?.max_file_upload_size_in_bytes) || 0;
      add(
        "token_valid",
        "Notion accepts the token",
        "pass",
        `Connected as \u201c${integrationName}\u201d${workspaceName ? ` in the \u201c${workspaceName}\u201d workspace` : ""}.`,
      );
    } else if (me.status === 401) {
      add(
        "token_valid",
        "Notion accepts the token",
        "fail",
        "Notion rejected the token. It has been revoked, the integration was deleted, or a space or quote mark was copied along with it.",
        tokenFix,
      );
    } else {
      add("token_valid", "Notion accepts the token", "fail", me.message, tokenFix);
    }
  } else {
    add("token_valid", "Notion accepts the token", "skip", "Nothing to test until NOTION_TOKEN is set.");
  }

  // --- 3. The database ---------------------------------------------------

  const rawId = (env.NOTION_DATABASE_ID || "").trim();
  const dbId = rawId.replace(/-/g, "");
  const idLooksRight = /^[0-9a-f]{32}$/i.test(dbId);
  const idFix =
    "Open the database in Notion as a full page and copy the address. The ID is the 32 letters and numbers after the last slash and before the ?v= \u2014 the part after ?v= identifies the view, not the database.";

  if (!rawId) {
    add("database_id", "Database ID has the right shape", "fail", "NOTION_DATABASE_ID is empty.", idFix);
  } else if (!idLooksRight) {
    add(
      "database_id",
      "Database ID has the right shape",
      "fail",
      `NOTION_DATABASE_ID is ${dbId.length} characters once dashes are removed; a Notion ID is 32. The whole link, the view ID or a page title is usually what got pasted.`,
      idFix,
    );
  } else {
    add("database_id", "Database ID has the right shape", "pass", "32 characters, correct shape.");
  }

  let schema = null;

  if (tokenWorks && idLooksRight) {
    const db = await notionProbe(env, `/databases/${dbId}`);
    if (db.ok) {
      const dbTitle = (db.data.title || []).map((t) => t.plain_text).join("") || "your database";
      add("database_access", "The integration can open the database", "pass", `Found \u201c${dbTitle}\u201d.`);

      const dsId = db.data.data_sources?.[0]?.id;
      if (!dsId) {
        add(
          "data_source",
          "The database has a data source",
          "fail",
          "Notion returned the database but no data source, which normally means this is a linked view rather than the real database.",
          "Use the ID of the original database, not of a linked or synced copy.",
        );
      } else {
        const ds = await notionProbe(env, `/data_sources/${dsId}`);
        if (ds.ok) {
          schema = ds.data.properties || {};
          add("data_source", "The database has a data source", "pass", "Schema read successfully.");
        } else {
          add("data_source", "The database has a data source", "fail", ds.message);
        }
      }
    } else if (db.status === 404) {
      add(
        "database_access",
        "The integration can open the database",
        "fail",
        `Your token works, but this database is invisible to \u201c${integrationName}\u201d. Nine times out of ten the integration has simply not been added to the page yet \u2014 this is the most common setup mistake by a wide margin.`,
        `In Notion, open the database as a full page \u2192 \u2022\u2022\u2022 menu, top right \u2192 Connections \u2192 Connect to \u2192 pick \u201c${integrationName}\u201d \u2192 Confirm. Then run this check again. If you have already done that, the ID belongs to a different database.`,
      );
    } else if (db.status === 400) {
      add(
        "database_access",
        "The integration can open the database",
        "fail",
        "Notion says this ID is not a database \u2014 most likely it is the ID of an ordinary page or of a view.",
        idFix,
      );
    } else {
      add("database_access", "The integration can open the database", "fail", db.message);
    }
  } else {
    add(
      "database_access",
      "The integration can open the database",
      "skip",
      "Skipped until the token and the ID above are both correct.",
    );
  }

  // --- 4. Properties -----------------------------------------------------

  if (schema) {
    const titleProp = pickProp(schema, "title", env.NOTION_TITLE_PROP);
    const filesProp = pickProp(schema, "files", env.NOTION_AUDIO_PROP);
    const tagsProp = pickProp(schema, "multi_select", env.NOTION_TAGS_PROP);
    const dateProp = pickProp(schema, "date", env.NOTION_DATE_PROP);
    const audioFlag = byName(schema, env.NOTION_AUDIO_FLAG_PROP || "Audio", "checkbox");
    const summaryProp = byName(schema, env.NOTION_SUMMARY_PROP || "Summary", "rich_text");

    add(
      "prop_title",
      "Title property",
      titleProp ? "pass" : "fail",
      titleProp ? `Notes are named in \u201c${titleProp}\u201d.` : "No title property found, so notes cannot be named.",
      titleProp ? "" : "Every Notion database has one. If you renamed it, set NOTION_TITLE_PROP to its name.",
    );

    add(
      "prop_files",
      "Attachment property",
      filesProp ? "pass" : "warn",
      filesProp
        ? `The recording is attached to \u201c${filesProp}\u201d.`
        : "No Files & media property. The recording still plays inside the note, but it will not show in a column.",
      filesProp ? "" : "Add a Files & media property, or set NOTION_AUDIO_PROP to the one you want used.",
      !filesProp,
    );

    add(
      "prop_tags",
      "Tags property",
      tagsProp ? "pass" : "warn",
      tagsProp ? `Tags you type go to \u201c${tagsProp}\u201d.` : "No multi-select property, so tags typed in the app are discarded.",
      tagsProp ? "" : "Add a Multi-select property named Tags, or set NOTION_TAGS_PROP.",
      !tagsProp,
    );

    add(
      "prop_summary",
      "Summary property",
      summaryProp ? "pass" : "warn",
      summaryProp
        ? `The one-line summary is copied into \u201c${summaryProp}\u201d.`
        : "No text property named Summary, so the summary appears only inside the note.",
      summaryProp ? "" : "Add a Text property named Summary, or set NOTION_SUMMARY_PROP.",
      !summaryProp,
    );

    add(
      "prop_date",
      "Date property",
      dateProp ? "pass" : "warn",
      dateProp
        ? `Recording time is written to \u201c${dateProp}\u201d.`
        : "No date property. Notion's own Created time still records when the note arrived.",
      "",
      !dateProp,
    );

    add(
      "prop_audio_flag",
      "Audio checkbox",
      audioFlag ? "pass" : "warn",
      audioFlag ? `\u201c${audioFlag}\u201d is ticked on every voice note.` : "No checkbox named Audio, so voice notes are not flagged.",
      "",
      !audioFlag,
    );
  }

  // --- 5. How long a recording this workspace will actually accept -------

  if (maxUpload) {
    const effective = Math.min(maxUpload, SINGLE_PART_MAX);
    const capped = maxUpload <= 5 * MIB;
    add(
      "file_limit",
      "Longest recording this workspace accepts",
      capped ? "warn" : "pass",
      capped
        ? `Your Notion plan caps uploads at ${Math.round(maxUpload / MIB)} MiB \u2014 roughly ${minutesFor(effective)} of speech. A longer note is transcribed and summarised correctly, then rejected by Notion when the audio is attached, and nothing is saved.`
        : `Your plan allows ${Math.round(maxUpload / MIB)} MiB per file. Audio is sent in one piece, so the practical ceiling is ${Math.round(effective / MIB)} MiB \u2014 roughly ${minutesFor(effective)} of speech.`,
      capped
        ? "Keep notes under that length, or upgrade the Notion workspace to a paid plan, which raises the limit to 5 GiB."
        : "",
      capped,
    );
  } else if (tokenWorks) {
    add(
      "file_limit",
      "Longest recording this workspace accepts",
      "skip",
      "Notion did not report a file size limit for this integration.",
      "",
      true,
    );
  }

  const failed = checks.filter((c) => c.status === "fail").length;

  return {
    ok: failed === 0,
    checkedAt: new Date().toISOString(),
    workspace: workspaceName,
    integration: tokenWorks ? integrationName : "",
    maxUploadBytes: maxUpload,
    summary: {
      passed: checks.filter((c) => c.status === "pass").length,
      warnings: checks.filter((c) => c.status === "warn").length,
      failed,
    },
    checks,
  };
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

    // A read-only checklist for a fresh install: passphrase, AI binding,
    // token, database, properties, upload limit. public/check.html renders it.
    if (url.pathname === "/api/check") {
      if (request.method === "OPTIONS") {
        return new Response(null, { status: 204, headers: corsHeaders(env) });
      }
      if (request.method !== "GET" && request.method !== "POST") {
        return json(env, 405, { ok: false, error: "Use GET" });
      }
      const given = request.headers.get("x-app-secret") || url.searchParams.get("secret") || "";
      if (env.APP_SECRET && given !== env.APP_SECRET) {
        return json(env, 401, {
          ok: false,
          error: "Unauthorized",
          hint: "That passphrase does not match APP_SECRET.",
        });
      }
      try {
        return json(env, 200, await runChecks(env));
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
