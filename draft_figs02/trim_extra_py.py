import shutil
from pathlib import Path
import os
import glob

for f in sorted(glob.glob("fig*.py.py")):
    src = Path(f)
    stem = src.stem
    new_name = f"{stem}"
    # check if src exists
    if src.exists():
        print(f"{src} --> {new_name}")
        os.rename(src, new_name)
    else:
        print(f"File {src} does not exist ---------------------------")
