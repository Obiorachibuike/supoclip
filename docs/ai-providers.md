# AI Providers and Processing Modes

SupoClip is free-first. A new installation can create an account, import or
upload a video, transcribe it, analyse it, and render subtitled clips with
face-centred framing **without an OpenAI, Gemini, Claude or AssemblyAI key**.

Two engines carry that free tier:

| Job | Engine | Cost |
|---|---|---|
| Clip reasoning (segments, hooks, titles, topics, summary) | **Agnes AI** (`agnes-2.5-flash`) | Free tier key |
| Transcription with word timestamps | **Faster-Whisper** (CTranslate2, local) | Free, runs on your CPU/GPU |
| Media work (cut, crop, resize, subtitles) | **FFmpeg** | Free |
| Face detection and reframing | **OpenCV / YuNet face model** (MediaPipe-compatible pipeline) | Free |

Paid providers remain available as upgrades and are never used silently.

## Choosing a mode

Set `AI_PROVIDER_MODE` to one of:

| Mode | Reasoning | Transcription | Paid keys needed |
|---|---|---|---|
| `free` (default) | Agnes AI, falling back to a local Ollama model when configured | Faster-Whisper | No |
| `premium` | Whatever `LLM` names (OpenAI, Gemini, Claude, OpenRouter) | `TRANSCRIPTION_PROVIDER` (AssemblyAI or local) | Yes, the matching key |
| `local` | Ollama (`OLLAMA_MODEL`) | Faster-Whisper or openai-whisper | No |
| `custom` | `LLM` as written | `TRANSCRIPTION_PROVIDER` as written | Only if you select one |

Leaving `AI_PROVIDER_MODE` blank keeps the behaviour an **existing** deployment
already had: SupoClip detects that you configured a provider (`LLM`,
`ASSEMBLY_AI_API_KEY`, `OPENAI_API_KEY`, …) and behaves like `custom`. Fresh
installs with no such variables start in `free`.

The frontend shows the active mode, the selected providers and whether a task
would spend money at **Settings → AI providers** (`/settings/ai`). Operators can
change the mode, keys and models from the admin runtime-settings screen as well
as from the environment.

## Free mode setup

1. Create a free API key at <https://platform.agnes-ai.com> (API Key page).
2. Add it to `.env`:

   ```env
   AI_PROVIDER_MODE=free
   AGNES_API_KEY=your_agnes_key
   ```

3. Start the stack (`docker-compose up -d`) and open Settings → AI providers to
   confirm `Agnes AI` is selected and `Faster-Whisper (local)` is active.

Optional free upgrades:

- `AGNES_MODEL` — defaults to `agnes-2.5-flash`.
- `OLLAMA_MODEL=gpt-oss:20b` — used by local mode and as the automatic free
  fallback if Agnes runs out of quota.
- `FASTER_WHISPER_MODEL` / `WHISPER_MODEL_SIZE` — `tiny`, `base`, `small`,
  `medium`, `large-v3`. Larger models are more accurate and slower; `medium` is
  the default.
- `FASTER_WHISPER_DEVICE=auto|cpu|cuda` and
  `FASTER_WHISPER_COMPUTE_TYPE=auto|int8|float16|float32`. `auto` detects CUDA
  and quietly falls back to CPU (`int8`).

## Premium mode setup

```env
AI_PROVIDER_MODE=premium
LLM=openai:gpt-5.2            # or google-gla:gemini-3-flash-preview, anthropic:claude-4-sonnet, openrouter:...
OPENAI_API_KEY=...
TRANSCRIPTION_PROVIDER=assemblyai   # optional; omit to keep local transcription
ASSEMBLY_AI_API_KEY=...
```

Selecting `premium` and supplying the key is the authorization. Paid usage is
then shown as **Paid provider in use** in the settings screen.

## What Agnes AI supports

SupoClip calls the documented Agnes endpoints only:

- Base URL `https://apihub.agnes-ai.com/v1`
- `POST /v1/chat/completions` with `Authorization: Bearer AGNES_API_KEY`
- `agnes-2.5-flash` is the default text model (`agnes-2.0-flash` is deprecated)
- Tool calling and streaming are supported; SupoClip uses tool calling for
  structured JSON output because native JSON-schema output is not documented
- Usage is reported in `usage.prompt_tokens` / `completion_tokens` /
  `total_tokens`

Documented error codes are mapped to actionable failures: `402` (quota or
balance exhausted) and `429` (rate limit) raise a quota error with guidance;
`401`/`403` name the key problem; `404` points at `AGNES_BASE_URL`/`AGNES_MODEL`;
`408`/`5xx` are retried and then reported as a temporary outage.

All Agnes traffic goes through `backend/src/agnes_ai.py`, so credentials,
timeouts, error translation and usage accounting live in one place. No endpoint
or model name is invented by SupoClip.

## Paid-provider protection

1. Free and local modes never *select* a paid provider. An explicit paid `LLM`
   or `TRANSCRIPTION_PROVIDER=assemblyai` is ignored (with a warning) unless
   `AI_ALLOW_PAID_PROVIDERS=true`.
2. Building a paid model without authorization raises
   `PaidProviderNotAuthorized` instead of making a request.
3. When Agnes runs out of quota or is unavailable, the pipeline first retries
   with the configured free local model (`OLLAMA_MODEL`). If there is none, the
   task stops with an explanation listing the exact ways forward. It never
   "just" falls back to a paid provider.
4. A local transcription failure never switches to AssemblyAI. The task fails
   with an explanation and the paid provider stays opt-in.

## Usage reporting

Providers report tokens per request. SupoClip records what they report per
process, logs it as an `ai_usage` line, and shows the counts in Settings → AI
providers, split by free and paid tiers. These are request and token counts, not
billing amounts; processes do not share counters and SupoClip does not estimate
costs.

## Caching

Reused across tasks and retries:

- Downloaded sources and uploaded originals (existing media cache)
- Extracted transcription audio and transcript word timings
  (`.transcript_cache.json`), shared with subtitle rendering
- Audio energy peaks (`.audio_peaks.json`), keyed by file identity and peak count
- Transcript analysis, stored per source and clip-length preset

Caches are per source file and storage path. Disable them with
`TRANSCRIPTION_CACHE_ENABLED=false`.

## Known limitations

- The free tier needs one free Agnes key for AI clip selection. Without it (and
  without Ollama) the pipeline stops with a configuration error — it does not
  guess clips silently.
- Faster-Whisper downloads model weights on first use; the first task on a new
  host is slower, and the host needs network access to Hugging Face or a
  pre-populated `FASTER_WHISPER_DOWNLOAD_ROOT`.
- Usage counters are per process (API pod vs ARQ workers), so the settings
  screen shows the API process's counters.
- Analysis caches are keyed by source and clip-length preset, not by provider;
  switching models intentionally reuses valid cached clip plans.
