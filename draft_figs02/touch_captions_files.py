from pathlib import Path
from base_names import base_names

for k, v in base_names.items():
    if v == 0:
        continue
    path1 = Path(k)
    # check if p exists
    if path1.with_suffix(".txt").exists():
        print(f"File already exists: {path1}")
    else:
        print(f"touching file: {path1.with_suffix('.txt')}")
        # p.with_suffix(".txt").touch()
