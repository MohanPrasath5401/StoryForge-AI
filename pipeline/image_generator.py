"""Keyframe generator: SDXL + IP-Adapter (character identity lock) with optional style LoRA."""
import gc, textwrap
from PIL import Image, ImageDraw, ImageOps
from utils.styles import NEGATIVE

KEY_W, KEY_H = 1280, 720

def _free():
    gc.collect()
    try:
        import torch; torch.cuda.empty_cache()
    except Exception:
        pass

class KeyframeGenerator:
    def __init__(self, cfg, mock=False):
        self.cfg, self.k = cfg, cfg["keyframe"]
        self.mock = mock or self.k["backend"] == "mock"
        self.pipe = None

    def load(self, log=print):
        if self.mock or self.pipe is not None:
            return
        import torch
        from diffusers import StableDiffusionXLPipeline
        if not torch.cuda.is_available():
            raise RuntimeError("No GPU found. In Colab: Runtime > Change runtime type > T4 GPU. (Or run with --mock to test without a GPU.)")
        log("Loading SDXL + IP-Adapter (first run downloads ~10 GB)...")
        pipe = StableDiffusionXLPipeline.from_pretrained(self.k["model"], torch_dtype=torch.float16, variant="fp16", use_safetensors=True)
        pipe.load_ip_adapter(self.k["ip_adapter_repo"], subfolder=self.k["ip_adapter_subfolder"], weight_name=self.k["ip_adapter_weight"])
        if self.k.get("lora_repo"):
            kw = {"weight_name": self.k["lora_weight_name"]} if self.k.get("lora_weight_name") else {}
            pipe.load_lora_weights(self.k["lora_repo"], adapter_name="style", **kw)
            pipe.set_adapters(["style"], adapter_weights=[self.k["lora_scale"]])
        pipe.enable_model_cpu_offload()
        pipe.enable_vae_slicing(); pipe.enable_vae_tiling()
        self.pipe = pipe

    def unload(self):
        self.pipe = None
        _free()

    def generate(self, prompt, ref_paths, out_path, seed, caption=""):
        if self.mock:
            return self._mock(prompt, ref_paths, out_path, seed, caption)
        import torch
        refs = [Image.open(p).convert("RGB") for p in ref_paths[:3]]
        if refs:
            self.pipe.set_ip_adapter_scale(self.k["ip_adapter_scale"])
            ip = [refs]
        else:  # no characters: neutralise adapter
            self.pipe.set_ip_adapter_scale(0.0)
            ip = [[Image.new("RGB", (224, 224), (128, 128, 128))]]
        gen = torch.Generator("cpu").manual_seed(seed)
        img = self.pipe(prompt=prompt, negative_prompt=NEGATIVE, ip_adapter_image=ip,
                        width=self.k["width"], height=self.k["height"],
                        num_inference_steps=self.k["steps"], guidance_scale=self.k["guidance"],
                        generator=gen).images[0]
        img = ImageOps.fit(img, (KEY_W, KEY_H), Image.LANCZOS)
        img.save(out_path)
        return out_path

    def _mock(self, prompt, ref_paths, out_path, seed, caption):
        img = Image.new("RGB", (KEY_W, KEY_H))
        d = ImageDraw.Draw(img)
        for y in range(KEY_H):
            t = y / KEY_H
            d.line([(0, y), (KEY_W, y)], fill=(int(240 - 60 * t), int(200 - 40 * t + seed % 30), int(160 + 40 * t)))
        n = max(len(ref_paths), 1)
        for i, p in enumerate(ref_paths[:3]):
            r = Image.open(p).convert("RGB").resize((380, 380))
            img.paste(r, (int((KEY_W - 380 * n - 60 * (n - 1)) / 2 + i * 440), 140))
        d.text((40, 40), textwrap.fill("MOCK " + caption, 90), fill=(30, 30, 30))
        img.save(out_path)
        return out_path
