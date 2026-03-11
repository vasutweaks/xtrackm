import os

for f in sorted(os.listdir(".")):
    if f.startswith("fig") and f.endswith(".py"):
        cmd = f"python3 {f}"
        print(cmd)
        try:
            os.system(cmd)
        except Exception as e:
            print(e)
            pass
