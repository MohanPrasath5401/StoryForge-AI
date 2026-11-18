"""Headless CLI:  python main.py --script_path examples/full_script_english.txt"""
import argparse, sys
from utils.config import load_config
from pipeline.runner import run_pipeline

def main():
    ap = argparse.ArgumentParser(description="StoryForge AI - script to video")
    ap.add_argument("--script_path", required=True)
    ap.add_argument("--project", default="my_story")
    ap.add_argument("--config", default=None)
    ap.add_argument("--backend", choices=["kenburns", "ltx"], default=None, help="animation backend")
    ap.add_argument("--style", choices=["storybook", "ghibli", "cinematic"], default=None)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--no_subtitles", action="store_true")
    ap.add_argument("--no_music", action="store_true")
    ap.add_argument("--lip_sync", action="store_true")
    ap.add_argument("--llm", action="store_true", help="refine visual prompts with a small LLM")
    ap.add_argument("--preview", action="store_true", help="keyframes only (cheap check of characters/look)")
    ap.add_argument("--fresh", action="store_true", help="delete cached project files first")
    ap.add_argument("--mock", action="store_true", help="no GPU/internet: placeholder visuals + silent audio (tests the plumbing)")
    a = ap.parse_args()
    cfg = load_config(a.config)
    text = open(a.script_path, encoding="utf-8").read()
    res = run_pipeline(text, cfg, project=a.project, mock=a.mock, backend=a.backend, subtitles=not a.no_subtitles,
                       music=not a.no_music, lip=a.lip_sync, use_llm=a.llm, preview=a.preview, fresh=a.fresh,
                       seed=a.seed, style=a.style)
    print("\nOutput:", res["video"] or res["project_dir"])

if __name__ == "__main__":
    sys.exit(main())
