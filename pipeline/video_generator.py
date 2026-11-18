"""Keyframe -> clip. 'kenburns' (FFmpeg, any hardware) or 'ltx' (LTX-Video image-to-video, GPU)."""
import gc, math, os
from utils.ffmpeg_utils import run

class VideoGenerator:
    def __init__(self, cfg, backend=None):
        self.cfg, self.v, self.a = cfg, cfg["video"], cfg["animation"]
        self.backend = backend or self.a["backend"]
        self.pipe = None

    # ---- normalisation: every clip ends up identical in codec/size/fps/length so concat is lossless
    def _normalize(self, src, out, dur):
        w, h, fps = self.v["width"], self.v["height"], self.v["fps"]
        vf = ("scale=%d:%d:force_original_aspect_ratio=increase,crop=%d:%d,fps=%d,"
              "tpad=stop_mode=clone:stop_duration=%.2f,format=yuv420p" % (w, h, w, h, fps, dur))
        run(["ffmpeg", "-y", "-i", src, "-vf", vf, "-t", "%.3f" % dur, "-an",
             "-c:v", "libx264", "-preset", "veryfast", "-crf", str(self.v["crf"]), out])

    def kenburns(self, image, out, dur, idx):
        w, h, fps = self.v["width"], self.v["height"], self.v["fps"]
        n = max(int(round(dur * fps)), 1)
        mode = idx % 4
        cx, cy = "iw/2-(iw/zoom/2)", "ih/2-(ih/zoom/2)"
        if mode == 0:   z, x, y = "1+0.14*on/%d" % n, cx, cy
        elif mode == 1: z, x, y = "1.14-0.14*on/%d" % n, cx, cy
        elif mode == 2: z, x, y = "1.12", "(iw-iw/zoom)*on/%d" % n, cy
        else:           z, x, y = "1.12", "(iw-iw/zoom)*(1-on/%d)" % n, cy
        vf = "scale=2880:-2,zoompan=z='%s':x='%s':y='%s':d=%d:s=%dx%d:fps=%d,format=yuv420p" % (z, x, y, n, w, h, fps)
        run(["ffmpeg", "-y", "-i", image, "-vf", vf, "-frames:v", str(n),
             "-c:v", "libx264", "-preset", "veryfast", "-crf", str(self.v["crf"]), out])

    def load(self, log=print):
        if self.backend != "ltx" or self.pipe is not None:
            return
        import torch
        from diffusers import LTXImageToVideoPipeline
        if not torch.cuda.is_available():
            raise RuntimeError("LTX needs a GPU. Use animation backend 'kenburns' on CPU.")
        dt = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        log("Loading LTX-Video (first run downloads ~10 GB)...")
        pipe = LTXImageToVideoPipeline.from_pretrained(self.a["ltx_model"], torch_dtype=dt)
        pipe.enable_model_cpu_offload()
        pipe.vae.enable_tiling()
        self.pipe = pipe

    def unload(self):
        self.pipe = None
        gc.collect()
        try:
            import torch; torch.cuda.empty_cache()
        except Exception:
            pass

    def generate(self, keyframe, out, dur, idx, prompt="", motion="", seed=0, tmp_dir="."):
        if self.backend == "kenburns":
            return self.kenburns(keyframe, out, dur, idx)
        import torch
        from PIL import Image, ImageOps
        from diffusers.utils import export_to_video
        W, H = self.a["ltx_width"], self.a["ltx_height"]
        img = ImageOps.fit(Image.open(keyframe).convert("RGB"), (W, H), Image.LANCZOS)
        p = "%s. %s. high quality, smooth natural motion" % (prompt, motion)
        frames = self.pipe(image=img, prompt=p[:900],
                           negative_prompt="worst quality, inconsistent motion, blurry, jittery, distorted, morphing faces",
                           width=W, height=H, num_frames=self.a["ltx_frames"],
                           num_inference_steps=self.a["ltx_steps"], guidance_scale=self.a["ltx_guidance"],
                           generator=torch.Generator("cpu").manual_seed(seed)).frames[0]
        raw = os.path.join(tmp_dir, "raw_%03d.mp4" % idx)
        export_to_video(frames, raw, fps=self.v["fps"])
        self._normalize(raw, out, dur)
        os.remove(raw)
        return out
