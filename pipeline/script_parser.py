"""Script -> list of ~5s narrative beats. Rule engine by default, optional small LLM to polish visual prompts."""
import json, math, re
from dataclasses import dataclass, field, asdict
from typing import List
from utils.styles import STYLES, DEFAULT_MOTION

TAMIL = re.compile(r"[\u0B80-\u0BFF]")

@dataclass
class Scene:
    index: int
    text: str
    lang: str
    characters: List[int] = field(default_factory=list)
    setting: str = ""
    visual_prompt: str = ""
    motion: str = DEFAULT_MOTION
    music: str = ""

    def to_dict(self):
        return asdict(self)

def detect_lang(text):
    letters = [c for c in text if c.isalpha()]
    if not letters:
        return "en"
    return "ta" if sum(1 for c in letters if TAMIL.match(c)) / len(letters) > 0.3 else "en"

def _split_words(text, budget):
    words = text.split()
    if len(words) <= budget:
        return [text]
    n = math.ceil(len(words) / budget)
    size = math.ceil(len(words) / n)
    return [" ".join(words[i:i + size]) for i in range(0, len(words), size)]

def _chunk(sentence, budget):
    """Split over-long sentences at clause boundaries (commas/semicolons/dashes) first, then by words."""
    if len(sentence.split()) <= budget:
        return [sentence]
    clauses = [c.strip() for c in re.split(r"(?<=[,;:\u2014])\s+", sentence) if c.strip()]
    out, cur = [], ""
    for c in clauses:
        if cur and len((cur + " " + c).split()) > budget:
            out.append(cur); cur = c
        else:
            cur = (cur + " " + c).strip()
    if cur:
        out.append(cur)
    return [p for o in out for p in _split_words(o, budget)]

def _beats(line, budget):
    sentences = [s.strip() for s in re.split(r"(?<=[.!?\u2026])\s+", line) if s.strip()]
    pieces = [p for s in sentences for p in _chunk(s, budget)]
    beats, cur = [], ""
    for p in pieces:
        if cur and len((cur + " " + p).split()) > budget:
            beats.append(cur); cur = p
        else:
            cur = (cur + " " + p).strip()
    if cur:
        beats.append(cur)
    return beats

def parse_script(text, cm, cfg, style=None) -> List[Scene]:
    """Script syntax:
       # comment
       @setting: rainy street market at dusk     (sticky until changed)
       @music: warm acoustic, gentle             (sticky)
       @motion: slow pan, drifting rain          (sticky)
       [06,17] Narration text...                 (sticky character tags)
    """
    style_txt = STYLES.get(style or cfg["keyframe"]["style"], STYLES["storybook"])
    max_chars = cfg["script"]["max_characters_per_scene"]
    wpb = cfg["script"]["words_per_beat"]
    setting, music, motion, tags = "", "", DEFAULT_MOTION, []
    scenes: List[Scene] = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        m = re.match(r"^@(setting|music|motion)\s*:\s*(.*)$", line, re.I)
        if m:
            k, v = m.group(1).lower(), m.group(2).strip()
            if k == "setting": setting = v
            elif k == "music": music = v
            else: motion = v or DEFAULT_MOTION
            continue
        m = re.match(r"^\[\s*([0-9,\s]+)\]\s*(.*)$", line)
        if m:
            tags = [int(x) for x in re.findall(r"\d+", m.group(1)) if int(x) in cm.characters]
            line = m.group(2).strip()
            if not line:
                continue
        mentioned = cm.detect(line)
        chars = (mentioned or tags)[:max_chars]
        if mentioned and tags:  # keep tagged cast, add mentioned ones if room
            chars = list(dict.fromkeys(tags + mentioned))[:max_chars]
        clean = cm.replace_mentions(line)
        lang = detect_lang(clean)
        for beat in _beats(clean, wpb.get(lang, 12)):
            sc = Scene(index=len(scenes), text=beat, lang=lang, characters=list(chars),
                       setting=setting, motion=motion, music=music)
            sc.visual_prompt = build_visual_prompt(sc, cm, style_txt)
            scenes.append(sc)
    return scenes

def build_visual_prompt(sc, cm, style_txt):
    cast = ", ".join(cm.get_prompt_description(c) for c in sc.characters)
    # English scene text only: Tamil narration is not understood by CLIP, so describe via setting/cast.
    action = sc.text if sc.lang == "en" else ""
    parts = [cast, action, sc.setting, style_txt]
    return ". ".join(p for p in parts if p)

def llm_refine(scenes, cm, cfg, log=print):
    """Optional: rewrite each beat into a concrete visual description with a small instruct LLM."""
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        name = cfg["script"]["llm_model"]
        tok = AutoTokenizer.from_pretrained(name)
        model = AutoModelForCausalLM.from_pretrained(name, torch_dtype=torch.float16, device_map="auto")
        for sc in scenes:
            cast = ", ".join(cm.get_prompt_description(c) for c in sc.characters) or "no named characters"
            msg = [{"role": "system", "content": "You write one-sentence English image prompts for a children's storybook illustration. Describe visible action, expression, setting and light. Max 35 words. No dialogue, no text."},
                   {"role": "user", "content": "Narration: %s\nSetting: %s\nCharacters: %s" % (sc.text, sc.setting or "unspecified", cast)}]
            ids = tok.apply_chat_template(msg, add_generation_prompt=True, return_tensors="pt").to(model.device)
            out = model.generate(ids, max_new_tokens=70, do_sample=False)
            desc = tok.decode(out[0][ids.shape[1]:], skip_special_tokens=True).strip().split("\n")[0]
            if desc:
                style_txt = STYLES.get(cfg["keyframe"]["style"], "")
                sc.visual_prompt = ". ".join(p for p in [cast, desc, style_txt] if p)
        del model
        import gc; gc.collect(); torch.cuda.empty_cache()
    except Exception as e:  # never block the pipeline on the optional LLM
        log("LLM refinement skipped (%s); using rule-based prompts." % e)
    return scenes

def save_scenes(scenes, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump([s.to_dict() for s in scenes], f, indent=2, ensure_ascii=False)
