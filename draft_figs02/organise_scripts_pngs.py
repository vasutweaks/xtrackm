import os
import re
from glob import glob
from base_names import base_names


def get_clean_name(filename):
    """Strips any existing 'figNN_' or 'supNN_' prefix."""
    fig_strip = re.sub(r'^(fig|sup)\d+_', '', filename)
    if fig_strip.endswith(".py"):
        ext_strip = fig_strip[:-3]
    elif fig_strip.endswith(".png"):
        ext_strip = fig_strip[:-4]
    elif fig_strip.endswith(".txt"):
        ext_strip = fig_strip[:-4]
    else:
        ext_strip = os.path.splitext(fig_strip)[0]
    return ext_strip


def strip_all_files():
    # files = sorted(glob(f"fig??_*.p*"))
    # strip all works only for fig|sup ending with .py or .png
    pattern = re.compile(r'^(fig|sup)\d+_.*\.(py|png|txt)$')
    files = [f for f in os.listdir('.') if pattern.match(f)]
    if not files:
        print("No matching files to strip.")
        return
    for f in files:
        if f.endswith(".png"):
            new_name = get_clean_name(f) + ".png"
            print(f"{f} -> {new_name}")
            os.rename(f, new_name)
        elif f.endswith(".py"):
            new_name = get_clean_name(f) + ".py"
            print(f"{f} -> {new_name}")
            os.rename(f, new_name)
        elif f.endswith(".txt"):
            new_name = get_clean_name(f) + ".txt"
            print(f"{f} -> {new_name}")
            os.rename(f, new_name)
        

def new_order_rename(base_names):
    # strip all works only for fig|sup ending with .py or .png
    strip_all_files()
    exts = (".py", ".png", ".txt")
    status_to_label = {1: "fig", 2: "sup"}
    counters = {"fig": 1, "sup": 1}

    def rename_group(original_key, label):
        idx = counters[label]
        target_stem = f"{label}{idx:02d}_{original_key}"
        renamed_any = False

        for ext in exts:
            old_name = f"{original_key}{ext}"
            if not os.path.exists(old_name):
                continue
            new_name = f"{target_stem}{ext}"
            print(f"{old_name} -> {new_name}")
            os.rename(old_name, new_name)
            renamed_any = True

        if renamed_any:
            counters[label] += 1

    print(f"{'Current File':<45} | {'New Name':<45} | {'Caption'}")
    print("-" * 60)

    for original_key, status in base_names.items():
        label = status_to_label.get(status)
        if label is None:
            continue
        rename_group(original_key, label)

# strip_all_files()
new_order_rename(base_names)
