"""Orchestrates the full pipeline with per-stage caching so interrupted Colab runs can resume."""
import json, math, os, shutil, time
from pipeline.character_manager import CharacterManager
from pipeline.script_parser import parse_script, save_scenes, llm_refine
from pipeline.image_generator import KeyframeGenerator
from pipeline.video_generator import VideoGenerator
from pipeline.voice_generator import VoiceGenerator
from pipeline.music_generator import MusicGenerator
from pipeline.lip_sync import lip_sync
from pipeline.assembler import assemble

def _safe(name):
    return "".join(c if c.isalnum() or c in "-_" else "_" for c in name.strip()) or "project"

def run_pipeline(script_text, cfg, project="my_story", mock=False, backend=None, subtitles=True,
                 music=True, lip=False, use_llm=False, preview=False, fresh=False, seed=None,
                 style=None, progress=lambda f, m: print("[%3d%%] %s" % (f * 100, m))):
    t0 = time.time()
    pdir = os.path.join(cfg["paths"]["outputs"], _safe(project))
    if fresh and os.path.isdir(pdir):
        shutil.rmtree(pdir)
    for sub in ("keyframes", "clips", "audio"):
        os.makedirs(os.path.join(pdir, sub), exist_ok=True)
    seed = cfg["keyframe"]["seed"] if seed is None else seed
    if style:
        cfg["keyframe"]["style"] = style
    fps, tail, min_s = cfg["video"]["fps"], cfg["video"]["tail_pad_seconds"], cfg["video"]["min_scene_seconds"]

    # 1. parse
    progress(0.02, "Parsing script into beats")
    cm = CharacterManager(cfg["paths"]["characters"])
    scenes = parse_script(script_text, cm, cfg)
    if not scenes:
        raise ValueError("Script is empty - nothing to generate.")
    if use_llm and not mock:
        scenes = llm_refine(scenes, cm, cfg, log=lambda m: progress(0.03, m))
    save_scenes(scenes, os.path.join(pdir, "scenes.json"))
    n = len(scenes)
    progress(0.05, "%d beats (~%.0f s of narration)" % (n, n * cfg["video"]["beat_seconds"]))

    # 2. narration first -> real durations drive scene length
    voice = VoiceGenerator(cfg, mock=mock)
    rows = []
    for i, sc in enumerate(scenes):
        wav = os.path.join(pdir, "audio", "narr_%03d.wav" % i)
        if not os.path.exists(wav):
            adur = voice.synth(sc.text, sc.lang, wav)
        else:
            from utils.ffmpeg_utils import duration
            adur = duration(wav)
        dur = max(min_s, adur + tail)
        dur = math.ceil(dur * fps) / fps
        rows.append({"index": i, "text": sc.text, "audio": wav, "audio_dur": adur, "duration": dur})
        progress(0.05 + 0.15 * (i + 1) / n, "Narration %d/%d" % (i + 1, n))

    # 3. keyframes (SDXL + IP-Adapter)
    kg = KeyframeGenerator(cfg, mock=mock)
    todo = [i for i in range(n) if not os.path.exists(os.path.join(pdir, "keyframes", "%03d.png" % i))]
    if todo:
        kg.load(log=lambda m: progress(0.2, m))
    for i, sc in enumerate(scenes):
        kf = os.path.join(pdir, "keyframes", "%03d.png" % i)
        rows[i]["keyframe"] = kf
        if i in todo:
            kg.generate(sc.visual_prompt, cm.ref_images(sc.characters), kf, seed + i, caption=sc.text)
        progress(0.2 + 0.3 * (i + 1) / n, "Keyframe %d/%d" % (i + 1, n))
    kg.unload()
    if preview:
        progress(1.0, "Preview done (keyframes only)")
        return {"project_dir": pdir, "keyframes": [r["keyframe"] for r in rows], "video": None}

    # 4. animation
    be = backend or cfg["animation"]["backend"]
    if mock and be == "ltx":
        be = "kenburns"
    vg = VideoGenerator(cfg, be)
    for i, sc in enumerate(scenes):
        clip = os.path.join(pdir, "clips", "%03d.mp4" % i)
        rows[i]["clip"] = clip
        if not os.path.exists(clip):
            vg.load(log=lambda m: progress(0.5, m))
            vg.generate(rows[i]["keyframe"], clip, rows[i]["duration"], i, sc.visual_prompt, sc.motion, seed + i, tmp_dir=pdir)
        if lip and sc.characters:
            synced = clip[:-4] + "_lip.mp4"
            if not os.path.exists(synced):
                lip_sync(clip, rows[i]["audio"], synced, cfg, log=lambda m: progress(0.8, m))
            if os.path.exists(synced):
                rows[i]["clip"] = synced
        progress(0.5 + 0.35 * (i + 1) / n, "Animated %d/%d (%s)" % (i + 1, n, be))
    vg.unload()

    # 5. music
    mwav = None
    if music and cfg["music"]["enabled"]:
        mwav = os.path.join(pdir, "audio", "music.wav")
        if not os.path.exists(mwav):
            mood = next((s.music for s in scenes if s.music), "")
            MusicGenerator(cfg, mock=mock).generate(mood, mwav, log=lambda m: progress(0.87, m))
        progress(0.9, "Music ready")

    # 6. assemble
    progress(0.92, "Assembling final video")
    final = assemble(pdir, rows, mwav, cfg, subtitles=subtitles, log=lambda m: progress(0.95, m))
    total = sum(r["duration"] for r in rows)
    progress(1.0, "Done: %.0f s video in %.1f min" % (total, (time.time() - t0) / 60))
    return {"project_dir": pdir, "video": final, "keyframes": [r["keyframe"] for r in rows], "seconds": total}
