"""28-character registry + IP-Adapter reference lookup (data-driven from assets/characters/characters.json)."""
import json, os, re
from typing import Dict, List, Optional

class CharacterManager:
    def __init__(self, characters_dir: str = "assets/characters"):
        self.characters_dir = characters_dir
        with open(os.path.join(characters_dir, "characters.json"), encoding="utf-8") as f:
            data = json.load(f)
        self.characters: Dict[int, Dict] = {c["id"]: c for c in data["characters"]}
        self.pairs: List[Dict] = data.get("pairs", [])
        for c in self.characters.values():
            folder = os.path.join(characters_dir, c["folder"])
            refs = [os.path.join(folder, n) for n in ("ref.jpg", "ref.png", "ref.jpeg") if os.path.exists(os.path.join(folder, n))]
            if not refs and os.path.isdir(folder):
                refs = [os.path.join(folder, f) for f in sorted(os.listdir(folder)) if f.lower().endswith((".png", ".jpg", ".jpeg")) and "full" not in f]
            c["images"] = refs

    def get(self, cid: int) -> Optional[Dict]:
        return self.characters.get(int(cid))

    def get_prompt_description(self, cid: int) -> str:
        c = self.get(cid)
        return "%s (%s)" % (c["name"], c["description"]) if c else ""

    def ref_images(self, cids: List[int]) -> List[str]:
        out = []
        for cid in cids:
            c = self.get(cid)
            if c and c["images"]:
                out.append(c["images"][0])
        return out

    def detect(self, text: str) -> List[int]:
        """Find characters mentioned as 'Character 06', 'char 6', or by display name (e.g. 'elderly chef')."""
        found = []
        low = text.lower()
        for m in re.finditer(r"\bchar(?:acter)?\s*0?(\d{1,2})\b", low):
            cid = int(m.group(1))
            if cid in self.characters and cid not in found:
                found.append(cid)
        for cid, c in self.characters.items():
            nm = re.sub(r"\s*\(.*?\)", "", c["name"]).lower()
            if len(nm) > 4 and re.search(r"\b%s\b" % re.escape(nm), low) and cid not in found:
                found.append(cid)
        return found

    def replace_mentions(self, text: str) -> str:
        """'Character 06' -> 'Elderly Chef' so narration and prompts read naturally."""
        def sub(m):
            c = self.get(int(m.group(1)))
            return re.sub(r"\s*\(.*?\)", "", c["name"]) if c else m.group(0)
        return re.sub(r"\b[Cc]har(?:acter)?\s*0?(\d{1,2})\b", sub, text)
