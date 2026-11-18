"""StoryForge AI web UI (Gradio): Simple mode, Advanced mode, Character gallery.   python app.py [--share]"""
import argparse, os
import gradio as gr
from utils.config import load_config
from pipeline.character_manager import CharacterManager
from pipeline.runner import run_pipeline

CFG = load_config()
CM = CharacterManager(CFG["paths"]["characters"])
EXAMPLE = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "examples", "full_script_english.txt"), encoding="utf-8").read()
BACKENDS = {"Fast - Ken Burns motion (works on any GPU, ~minutes)": "kenburns",
            "Cinematic - AI animation, LTX-Video (slow: ~2-4 min per 5s beat on T4)": "ltx"}

def generate(script, project, backend_label, style, subs, music, lip, llm, preview, mock, seed, progress=gr.Progress()):
    if not script.strip():
        raise gr.Error("Paste a script first.")
    try:
        res = run_pipeline(script, load_config(), project=project or "my_story", mock=mock,
                           backend=BACKENDS.get(backend_label, "kenburns"), subtitles=subs, music=music,
                           lip=lip, use_llm=llm, preview=preview, seed=int(seed), style=style,
                           progress=lambda f, m: progress(f, desc=m))
    except Exception as e:
        raise gr.Error(str(e))
    status = "Done. Files in: %s" % res["project_dir"]
    return res["video"], res["keyframes"], status

def gallery_items():
    return [(c["images"][0], "%02d %s" % (c["id"], c["name"])) for c in CM.characters.values() if c["images"]]

def pairs_md():
    rows = ["| Pair | Theme | Story ideas |", "|---|---|---|"]
    for p in CM.pairs:
        a, b = CM.get(p["a"]), CM.get(p["b"])
        rows.append("| %02d %s + %02d %s | %s | %s |" % (a["id"], a["name"], b["id"], b["name"], p["theme"], p["examples"]))
    return "\n".join(rows)

def build():
    with gr.Blocks(title="StoryForge AI", theme=gr.themes.Soft()) as demo:
        gr.Markdown("# StoryForge AI\nScript -> 3-5 minute illustrated video with 28 consistent characters. "
                    "Tag characters like `[06,17]` or write `Character 06`. Tamil and English supported.")
        with gr.Tab("Simple"):
            s_script = gr.Textbox(EXAMPLE, lines=14, label="Your script")
            s_mode = gr.Radio(list(BACKENDS), value=list(BACKENDS)[0], label="Quality")
            s_btn = gr.Button("Generate video", variant="primary")
            s_vid = gr.Video(label="Result")
            s_stat = gr.Markdown()
            s_gal = gr.Gallery(label="Keyframes", columns=4, height=300)
            s_btn.click(lambda sc, m: generate(sc, "my_story", m, "storybook", True, True, False, False, False, False, 1234),
                        [s_script, s_mode], [s_vid, s_gal, s_stat])
        with gr.Tab("Advanced"):
            a_script = gr.Textbox(EXAMPLE, lines=14, label="Script")
            with gr.Row():
                a_proj = gr.Textbox("my_story", label="Project name (re-run the same name to RESUME)")
                a_seed = gr.Number(1234, label="Seed", precision=0)
                a_style = gr.Dropdown(["storybook", "ghibli", "cinematic"], value="storybook", label="Style")
            a_mode = gr.Radio(list(BACKENDS), value=list(BACKENDS)[0], label="Animation")
            with gr.Row():
                a_subs = gr.Checkbox(True, label="Burn subtitles"); a_music = gr.Checkbox(True, label="Background music")
                a_lip = gr.Checkbox(False, label="Wav2Lip (experimental)"); a_llm = gr.Checkbox(False, label="LLM prompt refine")
                a_prev = gr.Checkbox(False, label="Preview keyframes only"); a_mock = gr.Checkbox(False, label="Mock mode (no GPU, test)")
            a_btn = gr.Button("Generate", variant="primary")
            a_vid = gr.Video(label="Result"); a_stat = gr.Markdown(); a_gal = gr.Gallery(label="Keyframes", columns=4, height=300)
            a_btn.click(generate, [a_script, a_proj, a_mode, a_style, a_subs, a_music, a_lip, a_llm, a_prev, a_mock, a_seed],
                        [a_vid, a_gal, a_stat])
        with gr.Tab("Characters"):
            gr.Gallery(gallery_items(), columns=7, height=520, label="28 characters (use the ID in [brackets])")
            gr.Markdown("### Recommended pairs\n" + pairs_md())
    return demo

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--share", action="store_true"); ap.add_argument("--port", type=int, default=7860)
    a = ap.parse_args()
    build().queue().launch(share=a.share, server_name="0.0.0.0", server_port=a.port)
