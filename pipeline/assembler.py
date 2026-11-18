"""FFmpeg assembly: concat clips, pad/concat narration, loop + duck music, burn subtitles, final 1080p24 MP4."""
import os
from utils.ffmpeg_utils import run, has_filter

def _ts(t):
    ms = int(round(t * 1000))
    h, ms = divmod(ms, 3600000); m, ms = divmod(ms, 60000); sec, ms = divmod(ms, 1000)
    return "%02d:%02d:%02d,%03d" % (h, m, sec, ms)

def write_srt(scenes, path):
    t, lines = 0.0, []
    for i, sc in enumerate(scenes, 1):
        end = t + min(sc["audio_dur"] + 0.15, sc["duration"])
        lines += [str(i), "%s --> %s" % (_ts(t), _ts(end)), sc["text"], ""]
        t += sc["duration"]
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

def _concat_list(files, path):
    with open(path, "w", encoding="utf-8") as f:
        for p in files:
            f.write("file '%s'\n" % os.path.abspath(p).replace("'", "'\\''"))

def assemble(project_dir, scenes, music_wav, cfg, subtitles=True, log=print, out_name="final.mp4"):
    """scenes: dicts with clip, audio, duration (video length), audio_dur, text."""
    v, mc, sc_cfg = cfg["video"], cfg["music"], cfg["subtitles"]
    work = os.path.join(project_dir, "work"); os.makedirs(work, exist_ok=True)
    total = sum(s["duration"] for s in scenes)

    # 1) video concat (all clips share codec/size/fps -> stream copy)
    _concat_list([s["clip"] for s in scenes], os.path.join(work, "clips.txt"))
    video = os.path.join(work, "video_all.mp4")
    run(["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", os.path.join(work, "clips.txt"), "-c", "copy", video])

    # 2) narration: pad each line to its scene length, then concat
    padded = []
    for i, s in enumerate(scenes):
        p = os.path.join(work, "narr_%03d.wav" % i)
        run(["ffmpeg", "-y", "-i", s["audio"], "-af", "apad", "-t", "%.3f" % s["duration"], "-ar", "44100", "-ac", "2", p])
        padded.append(p)
    _concat_list(padded, os.path.join(work, "narr.txt"))
    narr = os.path.join(work, "narration.wav")
    run(["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", os.path.join(work, "narr.txt"), "-c", "copy", narr])

    # 3) mix with ducking (music sidechained by narration) + loudness normalisation
    mix = os.path.join(work, "mix.wav")
    if music_wav and mc["enabled"]:
        fo = max(total - 2.5, 0)
        fc = ("[1:a]aformat=sample_rates=44100:channel_layouts=stereo,volume=%.2f,afade=t=in:d=1.5[m];"
              "[0:a]asplit=2[n1][n2];"
              "[m][n1]sidechaincompress=threshold=0.02:ratio=10:attack=30:release=600[duck];"
              "[n2][duck]amix=inputs=2:duration=first:normalize=0,afade=t=out:st=%.2f:d=2.5,loudnorm=I=-16:TP=-1.5:LRA=11[mix]") % (mc["volume"], fo)
        run(["ffmpeg", "-y", "-i", narr, "-stream_loop", "-1", "-i", music_wav, "-filter_complex", fc,
             "-map", "[mix]", "-t", "%.3f" % total, "-ar", "44100", mix])
    else:
        run(["ffmpeg", "-y", "-i", narr, "-af", "loudnorm=I=-16:TP=-1.5:LRA=11", "-ar", "44100", mix])

    # 4) subtitles + final mux
    srt = os.path.join(project_dir, "subtitles.srt")
    write_srt(scenes, srt)
    final = os.path.join(project_dir, out_name)
    burn = subtitles and sc_cfg["enabled"] and has_filter("subtitles")
    if subtitles and not burn:
        log("FFmpeg lacks libass; skipping burned-in subtitles (subtitles.srt still saved).")
    if burn:
        shutil_srt = os.path.join(work, "subs.srt")
        with open(srt, encoding="utf-8") as a, open(shutil_srt, "w", encoding="utf-8") as b:
            b.write(a.read())
        vf = "subtitles=work/subs.srt:force_style='FontName=%s,FontSize=%d,Outline=1.6,Shadow=0,MarginV=28'" % (sc_cfg["font"], sc_cfg["size"])
        run(["ffmpeg", "-y", "-i", "work/video_all.mp4", "-i", "work/mix.wav", "-vf", vf, "-map", "0:v", "-map", "1:a",
             "-c:v", "libx264", "-preset", "medium", "-crf", str(v["crf"]), "-pix_fmt", "yuv420p",
             "-c:a", "aac", "-b:a", "192k", "-shortest", "-movflags", "+faststart", out_name], cwd=project_dir)
    else:
        run(["ffmpeg", "-y", "-i", video, "-i", mix, "-map", "0:v", "-map", "1:a", "-c:v", "copy",
             "-c:a", "aac", "-b:a", "192k", "-shortest", "-movflags", "+faststart", final])
    return final
