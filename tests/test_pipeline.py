import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.config import load_config
from pipeline.character_manager import CharacterManager
from pipeline.script_parser import parse_script
from pipeline.runner import run_pipeline

def test_28_characters_have_refs():
    cm = CharacterManager(load_config()["paths"]["characters"])
    assert len(cm.characters) == 28
    assert all(c["images"] for c in cm.characters.values())

def test_parser_tags_and_mentions():
    cfg = load_config(); cm = CharacterManager(cfg["paths"]["characters"])
    sc = parse_script("[06,17] Character 06 smiled at the schoolgirl.\nNext line inherits tags.\nThe farmer waved.", cm, cfg)
    assert sc[0].characters[:2] == [6, 17]
    assert sc[1].characters == [6, 17]
    assert 22 in sc[2].characters and len(sc[2].characters) <= 3
    assert "Character 06" not in sc[0].text

def test_tamil_detected():
    cfg = load_config(); cm = CharacterManager(cfg["paths"]["characters"])
    assert parse_script("மழை பெய்யும் மாலை நேரம்.", cm, cfg)[0].lang == "ta"

def test_mock_end_to_end(tmp_path):
    cfg = load_config(overrides={"paths": {"outputs": str(tmp_path)}})
    r = run_pipeline("[06,17] A short test story. It has two beats.", cfg, project="ci", mock=True, progress=lambda f, m: None)
    assert os.path.getsize(r["video"]) > 10000
