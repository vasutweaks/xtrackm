import os
import re

pattern = re.compile(r'^(fig|sup)\d+_.*\.(py)$')
files = [f for f in os.listdir('.') if pattern.match(f)]

for f in files:
    cmd = f"MPLBACKEND=Agg python3 {f}"
    print(cmd)
    try:
        os.system(cmd)
    except Exception as e:
        print(e)
        pass
