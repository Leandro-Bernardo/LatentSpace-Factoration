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

    # training samples

        #rf"G:\Processar\MinIO\AlkalinityTrainingSamples",
        #rf"G:\Processar\MinIO\BisulfiteTrainingSamples",
        #rf"G:\Processar\MinIO\ChlorideTrainingSamples",
        #rf"G:\Processar\MinIO\EmulsionTrainingSamples",
        #rf"G:\Processar\MinIO\Iron2TrainingSamples",
        #rf"G:\Processar\MinIOetrobras\Iron3TrainingSamples",
        #rf"G:\Processar\MinIO\PhosphateTrainingSamples",
        #rf"G:\Processar\MinIO\pHTrainingSamples",
        #rf"G:\Processar\MinIOetrobras\RedoxTrainingSamples",
        #rf"G:\Processar\MinIO\SulfateTrainingSamples",
        #rf"G:\Processar\MinIO\SuspendedTrainingSamples",
        rf"G:\Dataset\[DAVID] Validar\adapted-jsons-type1"


    # inference samples
        #rf"G:\Processar\MinIOetrobras\AlkalinityInferenceSamples",
        #rf"G:\Processar\MinIOetrobras\BisulfiteInferenceSamples",
        #rf"G:\Processar\MinIO\ChlorideInferenceSamples",
        #rf"G:\Processar\MinIO\EmulsionInferenceSamples",
        #rf"G:\Processar\MinIO\Iron3InferenceSamples",
        #rf"G:\Processar\MinIO\PhosphateInferenceSamples",
        #rf"G:\Processar\MinIO\RedoxInferenceSamples",
        #rf"G:\Processar\MinIO\SulfateInferenceSamples",
        #rf"G:\Processar\MinIO\SuspendedInferenceSamples",

    ]


# Set intermediate dirs. for drive samples

    temp_dir = rf"G:\Dataset\Temporary"
    used_blanks_dir = rf"G:\Dataset\used_blanks"



# Set the results for drive samples

# drive training samples

    #result_dir = rf"G:\Dataset\Alkalinity\AlkalinityTrainingSamples"
    #result_dir = rf"G:\Dataset\Bisulfite\BisulfiteTrainingSamples"
    #result_dir = rf"G:\Dataset\Chloride\ChlorideTrainingSamples"
    #result_dir = rf"G:\Dataset\Phosphate\PhosphateTrainingSamples"
    #result_dir = rf"G:\Dataset\Emulsion\EmulsionTrainingSamples"
    #result_dir = rf"G:\Dataset\Iron2\Iron2TrainingSamples"
    #result_dir = rf"G:\Dataset\Iron3\Iron3TrainingSamples"
    #result_dir = rf"G:\Dataset\pH\pHTrainingSamples"
    #result_dir = rf"G:\Dataset\Phosphate\PhosphateTrainingSamples"
    #result_dir = rf"G:\Dataset\Redox\RedoxTrainingSamples"
    #result_dir = rf"G:\Dataset\Sulfate\SulfateTrainingSamples"
    #result_dir = rf"G:\Dataset\Suspended\SuspendedTrainingSamples"
    result_dir = fr'G:\Dataset\[DAVID] Validar\results'


# drive inference samples
    #result_dir = "G:\Dataset\Alkalinity\AlkalinityInferenceSamples"
    ###result_dir = "G:\Dataset\Bisulfite\BisulfiteInferenceSamples"
    #result_dir = "G:\Dataset\Chloride"#\ChlorideInferenceSamples"
    #result_dir = "G:\Dataset\Phosphate\PhosphateInferenceSamples"
    ###result_dir = "G:\Dataset\Emulsion\EmulsionInferenceSamples"
    ###result_dir = "G:\Dataset\Iron\IronInferenceSamples"
    #result_dir = "G:\Dataset\Phosphate\PhosphateInferenceSamples"
    ###result_dir = "G:\Dataset\Redox\RedoxInferenceSamples"
    #result_dir = "G:\Dataset\Sulfate\SulfateInferenceSamples"
    ###result_dir = "G:\Dataset\Suspended\SuspendedInferenceSamples"


    # Find all files.
    jsons, jpgs, dirs = all_files_and_dirs(base_dirs)
    # Move all files to the temp dir.
    os.makedirs(temp_dir, exist_ok=True)
    for key, (entry, path) in tqdm(jsons.items(), desc="Moving JSONs to the pool"):
        src_dir, json_filename = os.path.split(path)
        if not os.path.exists(os.path.join(temp_dir, json_filename)):
            shutil.move(os.path.join(src_dir, json_filename), os.path.join(temp_dir, json_filename))
        jsons[key][1] = os.path.join(temp_dir, json_filename)
    for key, path in tqdm(jpgs.items(), desc="Moving JPGs to the pool"):
        src_dir, jpg_filename = os.path.split(path)
        if not os.path.exists(os.path.join(temp_dir, jpg_filename)):
            shutil.move(os.path.join(src_dir, jpg_filename), os.path.join(temp_dir, jpg_filename))
        jpgs[key] = os.path.join(temp_dir, jpg_filename)
    # Delete empty dirs.
    for dir in tqdm(reversed(dirs), total=len(dirs), desc="Deleting empty dirs"):
        if len(os.listdir(dir)) == 0:
            os.rmdir(dir)
    # Fix files' location.
    missing_blanks = dict()
    for _, (entry, path) in tqdm(jsons.items(), desc="Building dataset"):
        # Skip blank samples.
        blank_filename = entry["sample"]["blankFileName"]
        if blank_filename is None:
            continue
        # Get basic fields.
        date = entry["sample"]["datetime"].split()[0].replace(".", "-")
        device_model = entry.get("device", UNKNOWN_SMARTPHONE_MODEL_DICT)["model"].lower().strip()
        analyst_name = entry["sample"]["analystName"].lower().strip()
        # Get reagent name.
        reagent_name = ""
        for reagent_field in ["colorReagent", "chlorideReagent", "liquidator", "phosphateReagent", "firstPhosphateReagent", "secondPhosphateReagent", "sulfateReagent","bisulfiteReagent","complexant", "acid","sulfateReagent", "sourceStock", "redoxIndicatorMix"]:
            if reagent_field in entry["sample"]:
                reagent_name = f"{reagent_name}, {entry['sample'][reagent_field]['name'].strip()}"
        reagent_name = reagent_name.lstrip(", ")
        if len(reagent_name) == 0:
            raise NotImplementedError
        # Get stock name.
        stock_name = None
        for source_stock_field in ["sourceStock", "alkalinitySourceStock", "chlorideSourceStock", "phosphateSourceStock", "sulfateSourceStock","sourceAliquot"]:
            if source_stock_field in entry["sample"]:
                stock_name = entry["sample"][source_stock_field]["name"]
                break
        if stock_name is None:
            raise NotImplementedError
        # Skip samples with missing blank sample.
        if blank_filename not in jpgs:
            if blank_filename in missing_blanks:
                missing_blanks[blank_filename][1] += 1
            else:
                missing_blanks[blank_filename] = [os.path.join(reagent_name, stock_name, f"{date} {device_model} ({analyst_name})"), 1]
            dst_dir = os.path.join(f"{result_dir}-MissingBlank", reagent_name, stock_name, f"{date} {device_model} ({analyst_name})")
        else:
            dst_dir = os.path.join(result_dir, reagent_name, stock_name, f"{date} {device_model} ({analyst_name})")
        # Make the destination dir.
        os.makedirs(dst_dir, exist_ok=True)
        os.makedirs(used_blanks_dir, exist_ok=True)
        # Move the JSON and JPG files of this sample to the dataset.
        src_dir, json_filename = os.path.split(path)
        for filename in [json_filename, entry["sample"]["fileName"], *entry["sample"].get("extraFileNames", [])]:
            if os.path.exists(os.path.join(src_dir, filename)) and not os.path.exists(os.path.join(dst_dir, filename)):
                shutil.move(os.path.join(src_dir, filename), os.path.join(dst_dir, filename))
        # Try to copy the blank sample.
        if blank_filename in jpgs:
            src_dir, jpg_filename = os.path.split(jpgs[blank_filename])
            json_filename = f"{os.path.splitext(jpg_filename)[0]}.json"
            # Try to copy the JSON and JPG files of the blank sample.
            if json_filename in jsons:
                entry, path = jsons[json_filename]
                jpg_filenames = [entry["sample"]["fileName"], *entry["sample"].get("extraFileNames", [])]
                for filename in [json_filename, *jpg_filenames]:
                    if os.path.exists(os.path.join(src_dir, filename)):
                        if src_dir != used_blanks_dir and not os.path.exists(os.path.join(used_blanks_dir, filename)):
                            shutil.move(os.path.join(src_dir, filename), os.path.join(used_blanks_dir, filename))
                        if used_blanks_dir != dst_dir and not os.path.exists(os.path.join(dst_dir, filename)):
                            shutil.copy(os.path.join(used_blanks_dir, filename), os.path.join(dst_dir, filename))
                # Update records.
                jsons[json_filename][1] = os.path.join(used_blanks_dir, json_filename)
                for jpg_filename in jpg_filenames:
                    jpgs[jpg_filename] = os.path.join(used_blanks_dir, jpg_filename)
            # Otherwise, copy at least the JPG file of the blank sample.
            else:
                if src_dir != used_blanks_dir and not os.path.exists(os.path.join(used_blanks_dir, jpg_filename)):
                    shutil.move(os.path.join(src_dir, jpg_filename), os.path.join(used_blanks_dir, jpg_filename))
                if used_blanks_dir != dst_dir and not os.path.exists(os.path.join(dst_dir, jpg_filename)):
                    shutil.copy(os.path.join(used_blanks_dir, jpg_filename), os.path.join(dst_dir, jpg_filename))
                # Update records.
                jpgs[jpg_filename] = os.path.join(used_blanks_dir, jpg_filename)
    # List missing blank samples.
    for blank_filename, (who, count) in sorted(missing_blanks.items()):
        print(f"Missing blank sample ({count} samples in '{who}' use {blank_filename})")


if __name__ == "__main__":
    main()
