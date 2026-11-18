# StoryForge AI

Open-source **script -> 3-5 minute illustrated video** engine with a built-in cast of **28 consistent characters**, bilingual (English/Tamil) narration, music with auto-ducking, and burned-in subtitles. $0 on free Colab GPUs.

```
script.txt -> beats -> narration (durations) -> SDXL+IP-Adapter keyframes -> motion (Ken Burns | LTX-Video)
           -> MusicGen score -> FFmpeg mix/duck/subtitles -> final.mp4 (1080p, 24 fps)
```

## Run it (3 ways)

**1. Google Colab (recommended).** Push this repo to GitHub, open `notebooks/StoryForge_Colab.ipynb` in Colab, set `REPO_URL`, pick **T4 GPU**, Run all. Open the `gradio.live` link.

**2. Local NVIDIA GPU (12 GB+).** Needs Python 3.10+, FFmpeg (`sudo apt install ffmpeg fonts-noto-core`).
`pip install -r requirements.txt && python app.py`

**3. CLI.** `python main.py --script_path examples/full_script_english.txt --project demo`

Test without any GPU/internet: add `--mock` (placeholder visuals, silent audio - validates the whole assembly chain).

## Writing scripts

```
@setting: a cozy rainy street market at dusk     # sticky visual setting
@music: warm acoustic, gentle piano               # sticky music mood
@motion: soft rain, slow push-in                  # sticky motion hint (LTX only)
[06,17] Character 06 smiled at the schoolgirl.    # [IDs] = cast, sticky until next tag
```
- Mention `Character 06` or the name ("elderly chef") and the character is auto-added. Max 3 per beat.
- Language is detected per beat (Tamil -> `ta-IN-PallaviNeural`, else `en-US-AvaNeural`).
- Beats are auto-split at clauses (~14 English / ~9 Tamil words ~ 5 s). **Scene length follows the real narration length**, so audio and picture never drift.
- Tamil beats: CLIP cannot read Tamil, so visuals come from cast + `@setting`. Always set `@setting` for Tamil scenes.
- Templates: `examples/`. Character IDs and pairings: **Characters** tab in the app or `assets/characters/characters.json`.

## Modes and real timings (Colab T4)

| Mode | Command | Cost |
|---|---|---|
| Preview keyframes | `--preview` | ~10-20 s per beat |
| **Fast** (default) | `--backend kenburns` | keyframes + FFmpeg; ~1-1.5 h for a 4-min film |
| Cinematic | `--backend ltx` | + ~2-4 min per beat for AI motion; a 4-min film (~48 beats) is 3-5 h, so it needs Drive saving and resume |

Runs are **resumable**: files are cached in `outputs/<project>/`. Re-run the same project name after a disconnect. Use `--fresh` to restart.

## Quality tips (consistency)
1. Always run `--preview` first; change `--seed` or the script until the keyframes look right.
2. Keep `ip_adapter_scale` 0.60-0.75 in `config.yaml`. Higher = faces lock but style distorts.
3. 1-2 characters per beat gives the best identity; 3 blends faces more (IP-Adapter limitation).
4. Style: `--style storybook|ghibli|cinematic`, or set `keyframe.lora_repo` to any SDXL style LoRA.
5. SDXL reads only the first 77 tokens: cast descriptors come first in the prompt on purpose.

## Corrections to the original architecture doc
- **`Wan2.1-I2V-1.3B` does not exist** (the 1.3B Wan model is text-to-video; image-to-video is 14B, too big for T4). Default AI motion is **LTX-Video**.
- **XTTS-v2 has no Tamil.** Edge-TTS is used. It is free but needs internet.
- **FLUX.1-schnell is not used for keyframes**: it has no mature IP-Adapter path for identity locking. SDXL + IP-Adapter is.
- **GitHub Actions has no GPU.** GitHub stores the code and runs a CPU smoke test (`.github/workflows/ci.yml`); video generation runs on Colab/local GPU.
- Subtitles are generated from the script and measured narration timings (exact, no Whisper needed).
- Wav2Lip is optional and **off by default**; it performs poorly on cartoon faces.

## Known limits
- `21_guitarist` source art has the face cropped at the top; regenerate that portrait for best identity.
- Characters 18/19 are the same boy; 15/16 and 05/27/12 are related designs - avoid mixing them in one beat.
- Hard cuts between beats (no crossfades) by design for narrative pacing.

## Layout
`app.py` UI | `main.py` CLI | `config.yaml` | `pipeline/` modules | `assets/characters/NN_name/ref.jpg` (IP-Adapter square crop) + `ref_full.jpg` | `notebooks/` | `tests/` | `examples/`

## Publish to GitHub
```
git init && git add . && git commit -m "StoryForge AI" 
git branch -M main && git remote add origin https://github.com/YOU/StoryForge-AI.git && git push -u origin main
```
