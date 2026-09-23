// Cloudflare Worker — receives a voice recording and files it in a Notion database.
// Audio arrives as the raw request body; metadata travels in headers.

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

async function transcribe(env, blob, filename) {
  if (!env.OPENAI_API_KEY) return "";
  const form = new FormData();
  form.append("file", blob, filename);
  form.append("model", env.OPENAI_TRANSCRIBE_MODEL || "whisper-1");
  const res = await fetch("https://api.openai.com/v1/audio/transcriptions", {
    method: "POST",
    headers: { Authorization: `Bearer ${env.OPENAI_API_KEY}` },
    body: form,
  });
  if (!res.ok) return ""; // transcription is optional — never block the upload
  const data = await res.json().catch(() => ({}));
  return (data.text || "").trim();
}

const chunk = (text, size = 1800) => text.match(new RegExp(`[\\s\\S]{1,${size}}`, "g")) || [];
const decodeHeader = (v) => {
  try {
    return decodeURIComponent(v || "").trim();
  } catch {
    return "";
  }
};

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
    transcribe(env, file, filename),
  ]);

  const ds = await notion(env, `/data_sources/${dataSourceId}`);
  const schema = ds.properties || {};
  const titleProp = pickProp(schema, "title", env.NOTION_TITLE_PROP);
  const filesProp = pickProp(schema, "files", env.NOTION_AUDIO_PROP);
  const dateProp = pickProp(schema, "date", env.NOTION_DATE_PROP);
  const tagsProp = pickProp(schema, "multi_select", env.NOTION_TAGS_PROP);

  // 1. reserve an upload slot, 2. send the bytes
  const upload = await notion(env, "/file_uploads", {
    method: "POST",
    body: { filename, content_type: mime },
  });
  const sendForm = new FormData();
  sendForm.append("file", file, filename);
  await notion(env, `/file_uploads/${upload.id}/send`, { method: "POST", form: sendForm });

  // 3. build the row
  const typedTitle = decodeHeader(request.headers.get("x-title"));
  const title =
    typedTitle ||
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

  const children = [
    { object: "block", type: "audio", audio: { type: "file_upload", file_upload: { id: upload.id } } },
  ];
  if (transcript) {
    children.push({
      object: "block",
      type: "heading_3",
      heading_3: { rich_text: [{ text: { content: "Transcript" } }] },
    });
    for (const part of chunk(transcript)) {
      children.push({
        object: "block",
        type: "paragraph",
        paragraph: { rich_text: [{ text: { content: part } }] },
      });
    }
  }

  const page = await notion(env, "/pages", {
    method: "POST",
    body: {
      parent: { type: "data_source_id", data_source_id: dataSourceId },
      icon: { type: "emoji", emoji: "🎙️" },
      properties,
      children,
    },
  });

  return json(env, 200, { ok: true, url: page.url, title, transcribed: Boolean(transcript) });
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
