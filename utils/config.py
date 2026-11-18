import copy, os
import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def _merge(a, b):
    for k, v in b.items():
        if isinstance(v, dict) and isinstance(a.get(k), dict):
            _merge(a[k], v)
        else:
            a[k] = v
    return a

def load_config(path=None, overrides=None):
    with open(os.path.join(ROOT, "config.yaml"), encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if path and os.path.abspath(path) != os.path.join(ROOT, "config.yaml"):
        with open(path, encoding="utf-8") as f:
            _merge(cfg, yaml.safe_load(f) or {})
    if overrides:
        _merge(cfg, copy.deepcopy(overrides))
    out = os.environ.get("STORYFORGE_OUTPUT")
    if out:
        cfg["paths"]["outputs"] = out
    for key in ("outputs", "characters"):
        p = cfg["paths"][key]
        cfg["paths"][key] = p if os.path.isabs(p) else os.path.join(ROOT, p)
    return cfg
