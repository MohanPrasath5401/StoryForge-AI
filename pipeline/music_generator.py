"""Ambient score via MusicGen-small (30s loop, looped/faded by the assembler)."""
import gc
from utils.ffmpeg_utils import run

class MusicGenerator:
    def __init__(self, cfg, mock=False):
        self.cfg, self.m, self.mock = cfg, cfg["music"], mock

    def generate(self, mood, out_wav, log=print):
        mood = mood or self.m["default_mood"]
        if self.mock:
            run(["ffmpeg", "-y", "-f", "lavfi", "-i", "sine=frequency=220:duration=30", "-af", "volume=0.2,tremolo=f=0.3:d=0.4", "-ar", "32000", out_wav])
            return out_wav
        import torch, scipy.io.wavfile as wf
        from transformers import AutoProcessor, MusicgenForConditionalGeneration
        log("Generating background music (%s)..." % mood)
        proc = AutoProcessor.from_pretrained(self.m["model"])
        model = MusicgenForConditionalGeneration.from_pretrained(self.m["model"]).to("cuda" if torch.cuda.is_available() else "cpu")
        inp = proc(text=[mood + ", seamless loop, no vocals"], padding=True, return_tensors="pt").to(model.device)
        audio = model.generate(**inp, do_sample=True, guidance_scale=3, max_new_tokens=1503)
        sr = model.config.audio_encoder.sampling_rate
        wf.write(out_wav, sr, audio[0, 0].cpu().numpy())
        del model; gc.collect(); torch.cuda.empty_cache()
        return out_wav
