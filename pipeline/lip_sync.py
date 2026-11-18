"""Optional Wav2Lip pass. Off by default: Wav2Lip is trained on real faces and is weak on cartoon art.
Setup: git clone https://github.com/Rudrabha/Wav2Lip third_party/Wav2Lip ; download wav2lip_gan.pth to checkpoints/."""
import os, subprocess

def lip_sync(video, audio, out, cfg, log=print):
    d, ck = cfg["lip_sync"]["wav2lip_dir"], cfg["lip_sync"]["checkpoint"]
    if not (os.path.isdir(d) and os.path.exists(ck)):
        log("Wav2Lip not installed - skipping lip sync for %s" % os.path.basename(video))
        return video
    r = subprocess.run(["python", "inference.py", "--checkpoint_path", os.path.abspath(ck), "--face", os.path.abspath(video),
                        "--audio", os.path.abspath(audio), "--outfile", os.path.abspath(out)], cwd=d, capture_output=True, text=True)
    if r.returncode != 0 or not os.path.exists(out):
        log("Wav2Lip failed (no face detected?) - keeping original clip.")
        return video
    return out
