// Cloudflare Worker — receives a voice recording and files it in a Notion database.
// Audio arrives as the raw request body; metadata travels in headers.
//
// Pipeline: auth → upload audio to Notion → transcribe (Workers AI) →
//           summarise (Workers AI) → create the row.
// Both AI steps are free, key-less and non-blocking: if either fails the note
// is still created with its audio attached.

const NOTION_API = "https://api.notion.com/v1";

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
    const out = await env.AI.run(model, input);
    return (out?.text || "").trim();
  } catch {
    return "";
  }
}

// ---- Summarisation (free, Cloudflare Workers AI) ------------------------
// Turns the raw transcript into a title, a short summary and action items.
// Set SUMMARY_ENABLED="false" to switch this off.

const SUMMARY_SCHEMA = {
  type: "object",
  properties: {
    title: { type: "string" },
    summary: { type: "string" },
    actions: { type: "array", items: { type: "string" } },
  },
  required: ["title", "summary", "actions"],
};

function summaryPrompt(env) {
  const language = env.SUMMARY_LANGUAGE
    ? `Write every field in ${env.SUMMARY_LANGUAGE}.`
    : "Write in the same language the speaker used.";
  return [
    "You clean up voice notes. The user speaks off the cuff, so the transcript rambles and may contain transcription errors.",
    "Return JSON only, with these keys:",
    '- "title": a specific headline of at most 8 words. No quotes, no trailing period.',
    '- "summary": 2-4 sentences covering what was actually said. Keep names, numbers and decisions. Never invent detail.',
    '- "actions": an array of short imperative tasks the speaker committed to. Use [] when there are none.',
    language,
  ].join("\n");
}

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
  const result = {
    title: clean(data.title).replace(/^["']|["'.]+$/g, "").slice(0, 120),
    summary: clean(data.summary),
    actions: Array.isArray(data.actions)
      ? data.actions.map(clean).filter(Boolean).slice(0, 25)
      : [],
  };
  return result.summary || result.actions.length ? result : null;
}

async function summarize(env, transcript) {
  if (!env.AI || !transcript) return null;
  if (String(env.SUMMARY_ENABLED).toLowerCase() === "false") return null;
  // A one-line note is already its own summary.
  const words = transcript.split(/\s+/).filter(Boolean).length;
  if (words < Number(env.SUMMARY_MIN_WORDS || 25)) return null;

  const model = env.SUMMARY_MODEL || "@cf/meta/llama-3.1-8b-instruct-fp8-fast";
  const messages = [
    { role: "system", content: summaryPrompt(env) },
    { role: "user", content: transcript.slice(0, Number(env.SUMMARY_MAX_CHARS || 12000)) },
  ];
  const base = { messages, max_tokens: 700, temperature: 0.2 };

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

// ---- Notion block helpers ----------------------------------------------

const chunk = (text, size = 1800) => text.match(new RegExp(`[\\s\\S]{1,${size}}`, "g")) || [];
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
const todo = (content) => ({
  object: "block",
  type: "to_do",
  to_do: { rich_text: rich(content.slice(0, 1800)), checked: false },
});

const decodeHeader = (v) => {
  try {
    return decodeURIComponent(v || "").trim();
  } catch {
    return "";
  }
};

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

  const mime = (request.headers.get("content-type") || "audio/webm").split(";")[0];
  const audio = await request.blob();
  if (!audio.size) throw new Error("Empty recording");

  const ext = EXT_BY_MIME[mime] || "weba";
  const stamp = new Date().toISOString().replace(/[:.]/g, "-").slice(0, 19);
  const filename = `voice-note-${stamp}.${ext}`;
  const file = new File([audio], filename, { type: mime });

  const [dataSourceId, transcript] = await Promise.all([
    resolveDataSourceId(env),
    transcribe(env, file),
  ]);

  // Reading the schema, pushing the bytes and summarising are independent.
  const [ds, upload, summary] = await Promise.all([
    notion(env, `/data_sources/${dataSourceId}`),
    uploadAudio(env, file, filename, mime),
    summarize(env, transcript),
  ]);

  const schema = ds.properties || {};
  const titleProp = pickProp(schema, "title", env.NOTION_TITLE_PROP);
  const filesProp = pickProp(schema, "files", env.NOTION_AUDIO_PROP);
  const dateProp = pickProp(schema, "date", env.NOTION_DATE_PROP);
  const tagsProp = pickProp(schema, "multi_select", env.NOTION_TAGS_PROP);
  const audioFlagProp = byName(schema, env.NOTION_AUDIO_FLAG_PROP || "Audio", "checkbox");
  const summaryProp = byName(schema, env.NOTION_SUMMARY_PROP || "Summary", "rich_text");

  // Title: what the user typed → what the model named it → first words → timestamp.
  const typedTitle = decodeHeader(request.headers.get("x-title"));
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
  const tags = decodeHeader(request.headers.get("x-tags"))
    .split(",")
    .map((t) => t.trim())
    .filter(Boolean);
  if (tagsProp && tags.length) properties[tagsProp] = { multi_select: tags.map((name) => ({ name })) };
  if (audioFlagProp) properties[audioFlagProp] = { checkbox: true };
  if (summaryProp && summary?.summary) {
    properties[summaryProp] = { rich_text: rich(summary.summary.slice(0, 2000)) };
  }

  // Page body: audio → summary → action items → transcript.
  const children = [
    { object: "block", type: "audio", audio: { type: "file_upload", file_upload: { id: upload.id } } },
  ];

  if (summary?.summary) {
    children.push(heading("Summary"));
    for (const part of chunk(summary.summary)) children.push(paragraph(part));
  }
  if (summary?.actions.length) {
    children.push(heading("Action items"));
    for (const item of summary.actions) children.push(todo(item));
  }
  if (transcript) {
    const body = chunk(transcript).map(paragraph);
    if (summary) {
      // Tuck the raw text away so the summary is what you read first.
      children.push({
        object: "block",
        type: "toggle",
        toggle: { rich_text: rich("Transcript"), children: body.slice(0, 95) },
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
    transcribed: Boolean(transcript),
    summarized: Boolean(summary),
  });
}

export default {
  async fetch(request, env) {
    const url = new URL(request.url);

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
