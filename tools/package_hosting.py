"""Stage only tracked public app files; local notebook files never enter Hosting."""
from pathlib import Path
import json
import shutil
import subprocess

ROOT = Path(__file__).resolve().parent.parent
PUBLIC_FILES = {
    "index.html", "app.js", "content.js", "core.js", "diagrams.js",
    "sources.js", "style.css", "themes.css", "themes.js", "study.js", "code-practice.js", "training-tools.js",
}


def main():
    config = json.loads((ROOT / "firebase.json").read_text(encoding="utf-8"))
    assert config["hosting"]["public"] == ".publish/site"
    output = ROOT / ".publish" / "site"
    for directory in [output.parent, output]:
        if directory.is_symlink() or (directory.exists() and directory.resolve() != directory):
            raise ValueError("Publish directory must stay inside the checkout")
        directory.mkdir(exist_ok=True)
    tracked = subprocess.check_output(
        ["git", "ls-files", "-z", "--", "dist/"], cwd=ROOT
    ).decode("utf-8").split("\0")
    files = {}
    for name in filter(None, tracked):
        relative = Path(name).relative_to("dist")
        vendor = relative.parts[0] == "vendor"
        if not (relative.as_posix() in PUBLIC_FILES or vendor):
            raise ValueError(f"Unapproved public file: {name}")
        if any(part.startswith(".") or part in {"local", "assets", "node_modules"}
               for part in relative.parts) or relative.suffix.lower() not in {
                   ".js", ".html", ".css", ".woff2", ".txt", ".md"
               }:
            raise ValueError(f"Private or unexpected file: {name}")
        source = ROOT / name
        if not source.is_file() or source.resolve() != source:
            raise ValueError(f"Missing file or symlink: {name}")
        files[relative] = source
    if not PUBLIC_FILES.issubset({p.as_posix() for p in files}):
        raise ValueError("Public app files are missing from Git")
    # Fail rather than ship stale files left by any earlier packaging process.
    for existing in output.rglob("*"):
        if existing.is_symlink() or existing.resolve() != existing:
            raise ValueError("Symlink in publish directory")
        if existing.is_file() and existing.relative_to(output) not in files:
            raise ValueError(f"Unexpected existing publish file: {existing.name}")
    for relative, source in files.items():
        destination = output / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
    size = sum(source.stat().st_size for source in files.values())
    print(f"Hosting package: {len(files)} public files, {size:,} bytes; no local notebooks")


if __name__ == "__main__":
    main()
