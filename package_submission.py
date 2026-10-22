"""Create the source-only ZIP accepted by the course server."""

import argparse
from pathlib import Path
import re
import zipfile


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("student_id")
    parser.add_argument("--include", nargs="*", default=[], help="Additional helper .py files")
    args = parser.parse_args()
    if not re.fullmatch(r"[A-Za-z0-9]{7}", args.student_id):
        parser.error("Use your own seven-character student ID")
    root = Path.cwd().resolve()
    files = sorted({Path("run.py"), Path("student_agent.py"), *map(Path, args.include)})
    for file in files:
        if (file.is_absolute() or ".." in file.parts or file.suffix != ".py"
                or file.is_symlink() or not (root / file).is_file()):
            parser.error(f"Expected a local Python source file: {file}")
    if sum((root / file).stat().st_size for file in files) > 100 * 1024:
        parser.error("Extracted Python source exceeds 100 KiB")
    destination = root / f"{args.student_id.lower()}.zip"
    with zipfile.ZipFile(destination, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for file in files:
            archive.write(root / file, file.as_posix())
    if destination.stat().st_size > 100 * 1024:
        destination.unlink()
        parser.error("ZIP exceeds 100 KiB")
    print(f"Created {destination.name}: {destination.stat().st_size} bytes. Upload it at https://arena.codefly.run")


if __name__ == "__main__":
    main()
