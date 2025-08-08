## Live Audio Transcription and Metrics

This document explains how the app captures speech, generates a live transcript, and provides live audio quality feedback. It includes a simple overview for non-technical readers and a detailed technical section with formulas and thresholds.

### For Everyone (Non‑Technical Overview)

- What happens when you press the mic:
  - The app starts recording your voice and shows a moving waveform so you know it’s listening.
  - Every second, a small piece of your audio is sent to the server and quickly turned into text. That text appears as a live transcript above the mic bar.
  - The app also monitors your audio level and gives a very short tip (a “hint”) when helpful, for example:
    - “Speak closer to the mic” when you’re too soft
    - “Reduce input volume” if you’re too loud
    - “Reduce background noise” if the environment is noisy
    - “Adjust speaking pace” if you’re talking too fast or slow

- Why the transcript and hints update smoothly instead of changing every split second:
  - We gently smooth the measurements and limit how often we update the screen. This avoids flicker and makes the experience calmer and easier to read.

- Privacy and performance:
  - Only short audio snippets (about 1 second each) are sent for live transcription while you’re recording.
  - If your internet is unstable, the transcript may update a bit slower.

### For Developers (Technical Details)

#### Data Flow

- Frontend (Next.js): `AudioInputComponent`
  - Uses `MediaRecorder(stream, { timeslice: 1000 })` to produce 1s audio chunks.
  - Each chunk is decoded via `AudioContext.decodeAudioData`, converted to 16‑bit PCM WAV, base64‑encoded, then POSTed to `POST /ai/transcribe/chunk`.
  - Live metrics are computed from time-domain samples read from an `AnalyserNode` (`getByteTimeDomainData`).
  - UI shows:
    - Rolling transcript (throttled, de‑duplicated)
    - Minimal hint above the mic bar

- Backend (FastAPI): `POST /ai/transcribe/chunk`
  - Request: `{ "audio_base64": string }` (WAV data)
  - Server decodes base64 and runs `quick_transcribe_chunk` (Whisper, fast settings, text only) and returns `{ text: string }`.

#### Frontend Metrics and Hints

Let \( x_i \) be the normalized audio sample at time index \( i \), where \( x_i \in [-1, 1] \). We obtain \( x_i \) by centering unsigned bytes around 128 and scaling: \( x_i = (b_i - 128) / 128 \).

- Root Mean Square (RMS) loudness
  - Formula: \( \mathrm{RMS} = \sqrt{\frac{1}{N} \sum_{i=1}^{N} x_i^2} \)
  - Interpretation: average loudness over the window.

- Peak level
  - Formula: \( \mathrm{Peak} = \max_i |x_i| \)
  - Interpretation: the highest absolute amplitude observed in the window.

- Voice Activity Heuristic and Speaking Rate (WPM)
  - We count “active” samples where \( |x_i| > 0.06 \). Let \( r = \frac{\text{activeSamples}}{N} \) be the activity ratio.
  - Speaking rate estimate (heuristic): \( \mathrm{WPM} \approx 90 + 100 \cdot r \)
  - Notes: This is a simple approximation meant for real‑time guidance, not a linguistic measurement.

- Exponential Moving Average (EMA) smoothing
  - We smooth RMS, Peak, and WPM with EMA to reduce jitter:
    - \( s_t = \alpha x_t + (1 - \alpha) s_{t-1} \)
  - Current \( \alpha = 0.1 \) (less sensitive to short spikes).

- UI throttling and debouncing
  - Metric UI updates are throttled to every ~300 ms.
  - Hint text changes are debounced to at least 1.8 s between changes.
  - Transcription requests are throttled to ≥ 1.5 s; repeated segments are ignored.

- Noise classification (simple heuristic)
  - Low noise: `RMS < 0.015` AND `Peak < 0.06`
  - High noise: `RMS > 0.12` AND `Peak > 0.3`
  - Otherwise: Moderate

- Hint logic (priority order)
  - If `Peak < 0.05`: “Speak closer to the mic”
  - Else if `Peak > 0.9`: “Reduce input volume”
  - Else if noise is High: “Reduce background noise”
  - Else if `WPM < 100` or `WPM > 200`: “Adjust speaking pace”
  - Else: no hint

#### Backend Chunk Transcription

- Endpoint: `POST /ai/transcribe/chunk`
  - Body: `{ "audio_base64": "..." }` (WAV 16‑bit PCM base64)
  - Decoding: base64 → bytes → temp wav file
  - Model: Whisper (local, cached)
  - Parameters optimized for latency (text only, no analysis):
    - `language="en"`, `task="transcribe"`
    - `word_timestamps=False`
    - `temperature=0.0`, `beam_size=1`, `best_of=1`
    - `condition_on_previous_text=False`
    - `no_speech_threshold=0.8`

#### Design Choices and Trade‑offs

- We favor smoothness and stability over absolute responsiveness (EMA + throttling) to avoid flicker.
- The WPM is a coarse heuristic sufficient for real‑time guidance; precise rates require word timings or NLP on the transcript.
- Noise classification is basic; a more robust approach could analyze spectral content and SNR, but this adds CPU cost.

#### Configuration and Requirements

- Environment
  - Frontend must have `NEXT_PUBLIC_BACKEND_URL` set to reach the API.
  - Browser support: `MediaRecorder` + Web Audio API (`AudioContext`) + secure context (HTTPS).

- Networking
  - Live transcript depends on short, periodic POSTs; poor connectivity may slow updates.

#### Future Enhancements

- Replace heuristic WPM with word-timing‑based estimates (from model word timestamps).
- WebSocket/SSE streaming for lower‑latency transcript updates.
- Better noise/SNR estimation and A‑weighted levels.
- Automatic gain control (AGC) guidance.

#### Glossary

- RMS: Root Mean Square, a standard measure of signal loudness.
- Peak: Maximum absolute amplitude observed over a window.
- EMA: Exponential Moving Average, smoothing technique \( s_t = \alpha x_t + (1-\alpha) s_{t-1} \).
- VAD: Voice Activity Detection, here a simple threshold on \( |x_i| \).
- WPM: Words Per Minute; here a heuristic based on the fraction of active samples.
- Throttle/Debounce: Rate-limiting techniques to stabilize UI updates.

