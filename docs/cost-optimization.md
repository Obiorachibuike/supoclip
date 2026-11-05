# Cost Optimization Guide

SupoClip can run with no per-request AI API charges when you provide your own compute. A hosted model or transcription API is easier to operate, but usage is metered by the provider. This guide describes configurations supported by the current source code.

## 1. Choose a cost profile

| Profile | Transcription | Clip analysis | API spend | Trade-off |
|---|---|---|---|---|
| Local-first | Local Whisper | Ollama | No per-request AI API fees | Uses your CPU/GPU and RAM; processing may be slower |
| Hybrid budget | AssemblyAI Universal-2 | Gemini Flash | Low, usage-based | Convenient cloud processing; provider charges still apply |
| Quality-first | AssemblyAI Universal-3.5 Pro | Your selected premium LLM | Higher, usage-based | Prioritizes provider quality over lowest cost |

“No per-request AI API fees” does not mean zero operating cost: local compute, electricity, storage, bandwidth, and hosting still cost money.

## 2. Local-first configuration

The repository already supports local Whisper transcription and Ollama model inference. No new API provider integration is required.

Example .env settings:

~~~env
# Transcription runs on the backend machine.
TRANSCRIPTION_PROVIDER=whisper
WHISPER_MODEL_SIZE=base
TRANSCRIPTION_LANGUAGE=

# Run transcript analysis locally through Ollama.
LLM=ollama:qwen3:8b
RERANK_LLM=ollama:qwen3:4b

# Keep paid download fallbacks disabled.
YOUTUBE_DOWNLOAD_PROVIDER=yt_dlp
APIFY_API_TOKEN=
RAPIDAPI_YOUTUBE_API_KEY=

# B-roll is optional; leave the key empty to avoid enabling the integration.
PEXELS_API_KEY=
~~~

Install Ollama on the machine that can be reached by the backend, then pull the models you choose. For example:

~~~bash
ollama pull qwen3:8b
ollama pull qwen3:4b
~~~

When the backend runs inside Docker and Ollama runs on the Docker host, the code's default Ollama URL is http://host.docker.internal:11434/v1. If Ollama runs elsewhere, set OLLAMA_BASE_URL to the reachable Ollama-compatible base URL. Do not expose an unauthenticated Ollama endpoint to the public internet.

The model names above are examples, not hardware guarantees. Select models that fit the available memory and validate structured JSON output quality with your own videos. A smaller model may be slower or less reliable on complex transcripts than a hosted model.

### Local Whisper trade-offs

The current whisper provider uses the openai-whisper package and requests word-level timestamps, which the caption renderer needs. base is a lower-resource starting point; small or medium may improve accuracy at the cost of more RAM and compute. The first run downloads the model weights, which are then cached by the local runtime.

The repository does not currently wire faster-whisper into its transcription provider. It is a promising future backend: its upstream project reports up to 4x faster inference than the reference Whisper implementation, with additional memory savings from 8-bit quantization. Treat this as a future integration, not an existing SupoClip feature.

## 3. Hybrid budget configuration

If you do not want to operate local inference, use the lower-cost transcription model and a cost-efficient hosted LLM:

~~~env
TRANSCRIPTION_PROVIDER=assemblyai
ASSEMBLY_AI_API_KEY=your_key
FAST_MODE_TRANSCRIPT_MODEL=universal-2
LLM=google-gla:gemini-3-flash-preview
GOOGLE_API_KEY=your_key
~~~

The fast preset now defaults to AssemblyAI universal-2. The implementation maps this model to a single Universal-2 selection rather than trying the Pro model first. It retains word timestamps and speaker labels required by the existing caption and speaker-framing pipeline.

To use Universal-2 for every processing mode, set:

~~~env
ASSEMBLY_AI_SPEECH_MODEL=universal-2
~~~

That global override takes precedence over the mode-specific transcription model. To opt back into the higher-end model for fast mode, set FAST_MODE_TRANSCRIPT_MODEL=universal-3-5-pro.

## 4. Keep downloads free-first

The default YouTube downloader is already yt_dlp. Keep these settings unless you intentionally want a paid fallback:

~~~env
YOUTUBE_DOWNLOAD_PROVIDER=yt_dlp
APIFY_API_TOKEN=
RAPIDAPI_YOUTUBE_API_KEY=
YOUTUBE_PROXY_URL=
~~~

The Apify and RapidAPI paths are optional integrations. Some Twitch and Kick provider configurations can invoke Apify when enabled, so review TWITCH_DOWNLOAD_PROVIDERS, KICK_DOWNLOAD_PROVIDERS, and APIFY_STREAM_PROVIDERS if those imports are part of your product.

Do not add a residential proxy just to reduce costs: proxy traffic is generally a separate paid service and does not guarantee successful downloads.

## 5. Reduce repeated AI work

SupoClip already has several reuse mechanisms:

- PostgreSQL processing_cache stores transcript text and analysis results.
- Per-task checkpoints preserve completed pipeline work across retries.
- Analysis cache entries are separated by clip-length preset.
- YouTube source artifacts and transcript metadata can be retained for later edits.

Avoid deleting cache rows or source metadata as part of routine cleanup unless you understand the effect on reprocessing. Reusing an existing source and analysis is generally cheaper than downloading, transcribing, and analysing it again.

The shared cache key includes the source URL, source type, processing mode, analysis cache version, and (where applicable) download quality. Different processing modes intentionally do not blindly share all analysis output.

## 6. Understand long-transcript LLM costs

The current analysis pipeline uses 12-minute transcript windows with 2-minute overlap. Long videos can therefore require multiple model calls, and multi-window analysis performs a final candidate reranking call. The default window concurrency is three.

For long videos:

- Prefer a model with good structured-output support at a sensible price.
- Use RERANK_LLM to select a less expensive model for the final comparison pass. Leave it unset if you want the primary LLM to handle reranking.
- Avoid enabling optional AI B-roll planning unless the output is worth the extra model calls and stock-footage requests.
- Keep clip counts and maximum source duration aligned with your actual product needs.
- Use the cache and task checkpoints rather than repeatedly creating fresh tasks for the same source and settings.

## 7. YouTube captions are a limited free option

The repository also supports:

~~~env
TRANSCRIPTION_PROVIDER=youtube_captions
~~~

This extracts available English YouTube captions through yt-dlp without a transcription API. However, the current implementation returns plain transcript text rather than word-level timestamps. It is not a drop-in replacement for AssemblyAI or Whisper when word-synchronized subtitles, speaker labels, and precise speaker framing are required. Use it only if that limitation is acceptable for your workflow.

## 8. Validate before deploying

After changing providers or models:

1. Restart or rebuild the backend and worker so they receive the new environment.
2. Process a short video with clear speech and at least two speakers.
3. Confirm transcript language, word timings, captions, and speaker framing.
4. Compare clip relevance and JSON validation failures between models.
5. Review logs for unexpected provider calls or paid fallbacks.
6. Test a long video to measure how many analysis windows and reranking calls are made.

Do not promise users unlimited free processing unless you have capacity limits. Local inference removes per-request model API charges, but compute, bandwidth, storage, and queue capacity remain finite.
