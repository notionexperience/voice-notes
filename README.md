# Voice → Notion

[![Deploy to Cloudflare](https://deploy.workers.cloudflare.com/button)](https://deploy.workers.cloudflare.com/?url=https://github.com/notionexperience/voice-notes)

Record a voice note in the browser and send it straight to a Notion database. The audio lands in a
Files property and plays inside the row; with an OpenAI key, a transcript is written into the page
body.

Free to run: GitHub for the code, Cloudflare Workers for hosting (100,000 requests/day on the free
plan, no credit card).

```
browser (public/index.html)  →  Worker (src/worker.js)  →  Notion API
     MediaRecorder                 holds the token          upload → create row
```

The Worker exists because browsers cannot call the Notion API directly (no CORS headers) and an
integration token must never sit in client-side code.

---

## Quick start

### 1. Notion token

Notion → `Settings` → `Connections` → enable **Developer mode** → **Personal access tokens** →
`+ New connection` → `â€¢â€¢â€¢` → **Copy internal connection token** (`ntn_â€¦`).

### 2. Target database

Any database works. Properties are matched by type, except the `Audio` checkbox, which is matched by
name so it cannot collide with other checkboxes:

| Property | Type | Used for |
| --- | --- | --- |
| Name | Title | note title â€” typed, or the first line of the transcript, or a timestamp |
| Attachments | Files | the recording |
| Recorded | Date | timestamp (optional) |
| Tags | Multi-select | tags typed in the recorder (optional) |
| Audio | Checkbox | ticked on every voice entry (optional) |

Missing properties are skipped. On the database: `â€¢â€¢â€¢` → **Connections** → add your connection.
Copy the 32-character ID from the URL, before `?v=`.

### 3. Deploy

Click the **Deploy to Cloudflare** button above. It copies this repository into your own GitHub
account, creates the Worker, and prompts for three values:

| Prompt | Value |
| --- | --- |
| `NOTION_TOKEN` | from step 1 |
| `NOTION_DATABASE_ID` | from step 2 |
| `APP_SECRET` | a long random string you invent â€” the password to your app |

About a minute later you get a URL like `https://voice-notes.<your-subdomain>.workers.dev`.
Every later push to the repo redeploys automatically.

<details>
<summary>Deploy from a terminal instead</summary>

```bash
git clone https://github.com/notionexperience/voice-notes.git
cd voice-notes
npm install
npx wrangler login
npx wrangler deploy

npx wrangler secret put NOTION_TOKEN
npx wrangler secret put NOTION_DATABASE_ID
npx wrangler secret put APP_SECRET
npx wrangler secret put OPENAI_API_KEY   # optional, enables transcription
```

Change `name` in `wrangler.jsonc` first â€” Worker names are unique per account. Secrets can also be
set in the dashboard: Worker → `Settings` → `Variables and Secrets` → **Add**.

</details>

### 4. Use it

Open the app once with the key:

```
https://voice-notes.<your-subdomain>.workers.dev/?key=<APP_SECRET>
```

The key is stored in that browser and removed from the address bar, so afterwards the plain URL
works on that device. Bookmark it, or add it to your phone's home screen.

- Big red button or **Space** starts and stops recording; **Pause** suspends it.
- **Upload**, or drag an audio file onto the card, works instead of recording.
- Tags: type and press Enter for each one.
- **Send to Notion** → a link to the new note appears.

In Notion you can paste the URL → **Create embed**, but embedded frames usually block the
microphone. Open the link in a browser tab for recording.

---

## Configuration

| Variable | Required | Notes |
| --- | --- | --- |
| `NOTION_TOKEN` | yes | secret |
| `NOTION_DATABASE_ID` | yes | the data source is resolved automatically |
| `APP_SECRET` | recommended | open the app as `?key=<APP_SECRET>`; without it anyone with the URL can post |
| `OPENAI_API_KEY` | no | enables transcription |
| `OPENAI_TRANSCRIBE_MODEL` | no | default `whisper-1` |
| `NOTION_TITLE_PROP` `NOTION_AUDIO_PROP` `NOTION_DATE_PROP` `NOTION_TAGS_PROP` `NOTION_AUDIO_FLAG_PROP` | no | property-name overrides |
| `NOTION_DATA_SOURCE_ID` | no | skip database lookup by passing the data source directly |
| `NOTION_VERSION` | no | default `2026-03-11` |
| `TIMEZONE` | no | used in fallback titles, e.g. `Europe/Berlin` |
| `ALLOWED_ORIGIN` | no | lock CORS to your domain when the page is hosted elsewhere |

Non-secret values can go in `wrangler.jsonc` under `vars`; secrets belong in Worker secrets.

---

## Google login instead of a key (optional)

Cloudflare Access puts a real sign-in in front of the Worker, including its `workers.dev` hostname.

1. Add Google as a login method: Zero Trust → **Integrations** → **Identity providers** → Google.
   You need a Google OAuth client whose redirect URI is
   `https://<team-name>.cloudflareaccess.com/cdn-cgi/access/callback`.
2. Workers & Pages → your Worker → **Access** tab → **Protect this Worker behind Access** →
   **All traffic** → add a policy allowing your email → **Apply Access**.
3. Optional: delete the `APP_SECRET` secret. The Worker skips the key check when it is absent, so
   the URL becomes clean.

Access login cannot complete inside a Notion embed â€” open the app in a tab.

---

## Transcription (optional)

Set `OPENAI_API_KEY` and the Worker adds a **Transcript** heading plus the text under the audio
block, and uses the first line as the title when you did not type one. Transcription never blocks
the upload: if it fails, the note is still created.

---

## Hosting the page on GitHub Pages instead

Optional, and only if you want the page on your own GitHub domain. GitHub Pages serves from the
repository root or `/docs` â€” not from `/public`.

1. Move `public/index.html` to `docs/index.html`.
2. Repo → `Settings` → `Pages` → deploy from `main`, folder `/docs`.
3. In `wrangler.jsonc`, remove the `assets` block; in `src/worker.js`, remove the
   `env.ASSETS.fetch(request)` fallback so the Worker serves the API only.
4. Set `ALLOWED_ORIGIN` to `https://<username>.github.io`.
5. Open the page with the API pointed at the Worker:
   `https://<username>.github.io/voice-notes/?api=https://voice-notes.<subdomain>.workers.dev/api/send-to-notion&key=<APP_SECRET>`

One host is simpler. The default setup keeps both on the Worker.

---

## Local development

```bash
cp .dev.vars.example .dev.vars   # fill in your values
npm install
npx wrangler dev
```

## Limits

- 20 MB per recording (Notion's single-part upload limit), roughly 30 minutes of speech.
- Free Notion workspaces cap file uploads at 5 MiB; paid plans at 5 GiB.
- Accepted audio: webm/opus, m4a, mp3, ogg, wav, aac, flac.

## Security

- The token lives only in Worker secrets â€” never in the repo, never in the browser.
- Set `APP_SECRET` or Cloudflare Access. Without either, anyone with the URL can write to your
  database. Neither lets anyone read from it: the Worker only creates pages.
- `.gitignore` excludes `.dev.vars`. If a token ever reaches a commit, rotate it in Notion.

## License

MIT - see [LICENSE](LICENSE).
