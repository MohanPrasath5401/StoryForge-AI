import subprocess

def run(cmd, cwd=None):
    p = subprocess.run(cmd, capture_output=True, text=True, cwd=cwd)
    if p.returncode != 0:
        raise RuntimeError("Command failed: %s\n%s" % (" ".join(map(str, cmd)), p.stderr[-2500:]))
    return p

def duration(path):
    p = run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", path])
    return float(p.stdout.strip())

def has_filter(name):
    p = subprocess.run(["ffmpeg", "-hide_banner", "-filters"], capture_output=True, text=True)
    return (" %s " % name) in p.stdout
