"""Bilingual narration. Edge-TTS (free, needs internet). Voice is chosen per beat from its language."""
import asyncio, os, threading
from utils.ffmpeg_utils import run, duration

class VoiceGenerator:
    def __init__(self, cfg, mock=False):
        self.cfg, self.c, self.mock = cfg, cfg["voice"], mock

    def voice_for(self, lang):
        return self.c["ta_voice"] if lang == "ta" else self.c["en_voice"]

    def _edge(self, text, voice, out):
        import edge_tts
        async def go():
            await edge_tts.Communicate(text, voice, rate=self.c["rate"]).save(out)
        err = []
        def runner():
            try: asyncio.run(go())
            except Exception as e: err.append(e)
        t = threading.Thread(target=runner); t.start(); t.join()   # safe inside Jupyter/Gradio loops
        if err: raise err[0]

    def synth(self, text, lang, out_wav):
        if self.mock:
            wps = 1.9 if lang == "ta" else 2.5
            d = max(len(text.split()) / wps, 1.0)
            run(["ffmpeg", "-y", "-f", "lavfi", "-i", "anullsrc=r=44100:cl=stereo", "-t", "%.2f" % d, out_wav])
            return duration(out_wav)
        tmp = out_wav[:-4] + ".mp3"
        last = None
        for _ in range(3):
            try:
                self._edge(text, self.voice_for(lang), tmp); break
            except Exception as e:
                last = e
        else:
            raise RuntimeError("Edge-TTS failed (needs internet): %s" % last)
        run(["ffmpeg", "-y", "-i", tmp, "-ar", "44100", "-ac", "2", out_wav])
        os.remove(tmp)
        return duration(out_wav)
