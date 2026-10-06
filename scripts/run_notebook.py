import argparse
import json
import re
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DONE = ("complete", "error", "cancel")
TIMING = re.compile(r"elapsed|seconds|minutes|projected")
SWITCH = "SMOKE = False"


def folder(name):
    path = ROOT / "notebooks" / name
    if not (path / "kernel-metadata.json").exists():
        sys.exit(f"no kernel-metadata.json in {path}")
    return path


def kernel_id(name):
    return json.loads((folder(name) / "kernel-metadata.json").read_text(encoding="utf-8"))["id"]


def build(name):
    source = folder(name) / f"{name}.py"
    if not source.exists():
        return
    subprocess.run([sys.executable, "-m", "jupytext", "--to", "ipynb", str(source)], check=True)


def push(name, smoke):
    build(name)
    notebook = folder(name) / f"{name}.ipynb"
    original = notebook.read_bytes()
    if smoke:
        if SWITCH.encode() not in original:
            sys.exit(f"{name} has no smoke switch")
        notebook.write_bytes(original.replace(SWITCH.encode(), b"SMOKE = True"))
    try:
        result = subprocess.run(["kaggle", "kernels", "push", "-p", str(folder(name))], capture_output=True, text=True)
    finally:
        if smoke:
            notebook.write_bytes(original)
    output = (result.stdout + result.stderr).strip()
    print(output)
    if result.returncode != 0 or "error" in output.lower():
        sys.exit(f"push of {name} failed")


def wait(name, poll):
    kid = kernel_id(name)
    while True:
        result = subprocess.run(["kaggle", "kernels", "status", kid], capture_output=True, text=True)
        status = (result.stdout + result.stderr).strip()
        print(time.strftime("%H:%M:%S"), status, flush=True)
        if any(word in status.lower() for word in DONE):
            return
        time.sleep(poll)


def fetch(name, pattern):
    dest = ROOT / "kaggle_out" / name
    dest.mkdir(parents=True, exist_ok=True)
    current = dest / "stdout.txt"
    if current.exists():
        current.replace(dest / "stdout_prev.txt")
    command = ["kaggle", "kernels", "output", kernel_id(name), "-p", str(dest), "-o"]
    if pattern:
        command += ["--file-pattern", pattern]
    subprocess.run(command, check=True)
    for log in dest.glob("*.log"):
        entries = json.loads(log.read_text(encoding="utf-8"))
        for stream in ("stdout", "stderr"):
            text = "".join(e["data"] for e in entries if e.get("stream_name") == stream)
            (dest / f"{stream}.txt").write_text(text, encoding="utf-8")
    print(f"fetched into {dest}")


def compare(name):
    dest = ROOT / "kaggle_out" / name
    before = [l for l in (dest / "stdout_prev.txt").read_text(encoding="utf-8").splitlines() if not TIMING.search(l)]
    after = [l for l in (dest / "stdout.txt").read_text(encoding="utf-8").splitlines() if not TIMING.search(l)]
    changed = [(a, b) for a, b in zip(before, after) if a != b]
    if len(before) != len(after):
        print(f"line count differs: {len(before)} vs {len(after)}")
    for a, b in changed:
        print(f"- {a}\n+ {b}")
    print("identical" if not changed and len(before) == len(after) else f"{len(changed)} lines differ")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["build", "push", "wait", "fetch", "compare"])
    parser.add_argument("name")
    parser.add_argument("--poll", type=int, default=120)
    parser.add_argument("--pattern", default=None)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if args.action == "build":
        build(args.name)
    elif args.action == "push":
        push(args.name, args.smoke)
    elif args.action == "wait":
        wait(args.name, args.poll)
    elif args.action == "fetch":
        fetch(args.name, args.pattern)
    else:
        compare(args.name)


if __name__ == "__main__":
    main()
