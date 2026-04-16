from tqdm import tqdm
from typing import Any, Dict, List, Tuple
import json, os, shutil


def all_files_and_dirs(base_dirs: List[str]) -> Tuple[Dict[str, Any], Dict[str, str], List[str]]:
    jsons: Dict[str, Any] = dict()  # {json_filename: [sample, path]}
    jpgs: Dict[str, str] = dict()   # {jpg_filename: path}
    dirs: List[str] = list()        # [dir, ...]
    not_visited_dirs = list(base_dirs)
    with tqdm(desc="Scanning folders", total=len(base_dirs)) as pbar:
        while len(not_visited_dirs) != 0:
            current_dir = not_visited_dirs.pop(0)
            dirs.append(current_dir)
            for filename in sorted(os.listdir(current_dir)):
                path = os.path.join(current_dir, filename)
                if os.path.isfile(path):
                    if filename.lower().endswith(".json"):
                        with open(path, "r", encoding="utf8") as file:
                            entry = json.load(file)
                            jsons[filename] = [entry, path]
                    elif filename.lower().endswith(".jpg"):
                        jpgs[filename] = path
                elif os.path.isdir(path):
                    not_visited_dirs.append(path)
                    pbar.total += 1
                    pbar.refresh()
            pbar.update(1)
    return jsons, jpgs, dirs


def main() -> None:
    UNKNOWN_SMARTPHONE_MODEL_DICT = {"model": "unknown"}
    # Set the base dirs.
    base_dirs = [
        "/mnt/c/Users/laffe/Desktop/Samples/Current/Temp",
    ]
    # Set the result dir.
    result_dir = "/mnt/c/Users/laffe/Desktop/xyz"
    # Find all files.
    jsons, jpgs, dirs = all_files_and_dirs(base_dirs)
    # Fix files' location.
    for _, (entry, path) in tqdm(jsons.items(), desc="Building dataset"):
        # Skip non blank samples.
        blank_filename = entry["sample"]["blankFileName"]
        if blank_filename is not None:
            continue
        # Get basic fields.
        date = entry["sample"]["datetime"].split()[0].replace(".", "-")
        device_model = entry.get("device", UNKNOWN_SMARTPHONE_MODEL_DICT)["model"].lower().strip()
        analyst_name = entry["sample"]["analystName"].lower().strip()
        # Get reagent name.
        reagent_name = ""
        for reagent_field in ["colorReagent", "chlorideReagents", "liquidator", "phosphateReagent", "firstPhosphateReagent", "secondPhosphateReagent", "sulfateReagent"]:
            if reagent_field in entry["sample"]:
                reagent_name = f"{reagent_name}, {entry['sample'][reagent_field]['name'].strip()}"
        reagent_name = reagent_name.lstrip(", ")
        if len(reagent_name) == 0:
            raise NotImplementedError
        # Make the destination dir.
        dst_dir = os.path.join(result_dir, reagent_name, f"{date} {device_model} ({analyst_name})")
        os.makedirs(dst_dir, exist_ok=True)
        # Move the JSON and JPG files of this sample to the dataset.
        src_dir, json_filename = os.path.split(path)
        for filename in [json_filename, entry["sample"]["fileName"], *entry["sample"].get("extraFileNames", [])]:
            if os.path.exists(os.path.join(src_dir, filename)) and not os.path.exists(os.path.join(dst_dir, filename)):
                shutil.move(os.path.join(src_dir, filename), os.path.join(dst_dir, filename))


if __name__ == "__main__":
    main()
