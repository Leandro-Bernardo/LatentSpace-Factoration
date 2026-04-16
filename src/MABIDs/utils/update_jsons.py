from argparse import ArgumentParser, Namespace
from tqdm import tqdm
from typing import Any, Dict, Final, List, Optional, Tuple
import json, os


DEFAULT_BASE_DIRS = [
    "/media/prograf/DISK1/Processar/Dataset_03_outubro/Alkalinity/",
    "/media/prograf/DISK1/Processar/Dataset_03_outubro/Chloride/",
    "/media/prograf/DISK1/Processar/Dataset_03_outubro/Phosphate/",
    "/media/prograf/DISK1/Processar/Dataset_03_outubro/Sulfate/",

     #os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "Alkalinity")),
     #os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "Chloride")),
    # os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "MABIDs-Dataset-Experimental")),
     #os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "Phosphate")),
     #os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "Sulfate")),
]


ANALYTE_TYPES: Final[List[str]] =["alkalinity", "chloride", "phosphate", "sulfate","iron3"]


def add_if_does_not_exist(entry: Dict[str, Any], keys: Tuple[str, ...], value: Any) -> None:
    for key in keys[:-1]:
        if key not in entry:
            entry[key] = dict()
        entry = entry[key]
    if keys[-1] not in entry:
        entry[keys[-1]] = value


def delete_if_exists(entry: Dict[str, Any], keys: Tuple[str, ...]) -> None:
    for key in keys[:-1]:
        if key not in entry:
            return
        entry = entry[key]
    if keys[-1] in entry:
        del entry[keys[-1]]


def rename_if_exists(entry: Dict[str, Any], from_keys: Tuple[str, ...], to_keys: Tuple[str, ...]) -> None:
    value = entry
    for key in from_keys:
        if key not in value:
            return
        value = value[key]
    add_if_does_not_exist(entry, to_keys, value)
    delete_if_exists(entry, from_keys)


def old_jsons(base_dirs: List[str]) -> List[Tuple[Dict[str, Any], str, str]]:
    result: List[Tuple[Dict[str, Any], str, str]] = list()  # List[Tuple[json, path, analyte]]
    dirs = list(base_dirs)
    with tqdm(desc="Scanning folders", total=len(dirs)) as pbar_dirs:
        while len(dirs) != 0:
            current_dir = dirs.pop(0)
            for filename in sorted(os.listdir(current_dir)):
                path = os.path.join(current_dir, filename)
                if os.path.isfile(path) and filename.lower().endswith(".json"):
                    analyte: Optional[str] = None
                    for analyte_type in ANALYTE_TYPES:
                        if filename.lower().startswith(analyte_type):
                            analyte = analyte_type
                            break
                    if analyte is None:
                        raise RuntimeError("The analyte type could not be identified.")
                    with open(path, "r", encoding="utf8") as file:
                        result.append((json.load(file), path, analyte))
                elif os.path.isdir(path):
                    dirs.append(path)
                    pbar_dirs.total += 1
                    pbar_dirs.refresh()
            pbar_dirs.update(1)
    return result


def main(args: Namespace) -> None:
    # Find files to fix.
    for entry, path, analyte in tqdm(old_jsons(args.base_dirs), desc="Updating samples"):
        # Remove fields.
        delete_if_exists(entry, ("sample", "spectrophotometerFile"))
        # Rename fields.
        rename_if_exists(entry, ("app", "package_name"), ("app", "packageName"))
        rename_if_exists(entry, ("app", "app_name"), ("app", "appName"))
        rename_if_exists(entry, ("app", "version_name"), ("app", "versionName"))
        rename_if_exists(entry, ("device", "android_version"), ("device", "androidVersion"))
        rename_if_exists(entry, ("sample", f'{analyte}SourceStock'), ("sample", "sourceStock"))
        rename_if_exists(entry, ("sample", "sourceStock", "concentration"), ("sample", "sourceStock", f'{analyte}Concentration'))
        rename_if_exists(entry, ("sample", "sourceStock", "concentrationUnity"), ("sample", "sourceStock", f'{analyte}ConcentrationUnit'))
        rename_if_exists(entry, ("sample", "sourceStock", "alkalinityUnity"), ("sample", "sourceStock", "alkalinityUnit"))
        rename_if_exists(entry, ("sample", "sourceStock", "chlorideConcentrationUnity"), ("sample", "sourceStock", "chlorideConcentrationUnit"))
        rename_if_exists(entry, ("sample", "sourceStock", "phosphateConcentrationUnity"), ("sample", "sourceStock", "phosphateConcentrationUnit"))
        rename_if_exists(entry, ("sample", "sourceStock", "sulfateConcentrationUnity"), ("sample", "sourceStock", "sulfateConcentrationUnit"))
        rename_if_exists(entry, ("sample", f'{analyte}SourceAliquot'), ("sample", "sourceAliquot"))
        rename_if_exists(entry, ("sample", "sourceAliquot", "aliquotUnity"), ("sample", "sourceAliquot", "aliquotUnit"))
        rename_if_exists(entry, ("sample", "sourceAliquot", "finalVolumeUnity"), ("sample", "sourceAliquot", "finalVolumeUnit"))
        rename_if_exists(entry, ("sample", "phosphateReagent"), ("sample", "firstPhosphateReagent"))
        rename_if_exists(entry, ("sample", "alkalinityUnity"), ("sample", "alkalinityUnity"))
        rename_if_exists(entry, ("sample", "concentrationUnity"), ("sample", "concentrationUnit"))
        rename_if_exists(entry, ("sample", "volumeUnity"), ("sample", "volumeUnit"))
        for aliquot in entry["sample"]["sourceStock"]["aliquots"]:
            rename_if_exists(aliquot, ("aliquotUnity",), ("aliquotUnit",))
            rename_if_exists(aliquot, ("finalVolumeUnity",), ("finalVolumeUnit",))
        # Add fields.
        if analyte == "alkalinity":
            add_if_does_not_exist(entry, ("sample", "colorReagent", "instructionsEnUs"), "")
            add_if_does_not_exist(entry, ("sample", "colorReagent", "instructionsPtBr"), "")
        elif analyte == "chloride":
            add_if_does_not_exist(entry, ("sample", "chlorideReagent", "instructionsEnUs"), "")
            add_if_does_not_exist(entry, ("sample", "chlorideReagent", "instructionsPtBr"), "")
            add_if_does_not_exist(entry, ("sample", "liquidator", "instructionsEnUs"), "")
            add_if_does_not_exist(entry, ("sample", "liquidator", "instructionsPtBr"), "")
        elif analyte == "phosphate":
            add_if_does_not_exist(entry, ("sample", "firstPhosphateReagent", "instructionsEnUs"), "")
            add_if_does_not_exist(entry, ("sample", "firstPhosphateReagent", "instructionsPtBr"), "")
            add_if_does_not_exist(entry, ("sample", "secondPhosphateReagent", "name"), "Unknown")
            add_if_does_not_exist(entry, ("sample", "secondPhosphateReagent", "components"), [])
            add_if_does_not_exist(entry, ("sample", "secondPhosphateReagent", "instructionsEnUs"), "")
            add_if_does_not_exist(entry, ("sample", "secondPhosphateReagent", "instructionsPtBr"), "")
        elif analyte == "sulfate":
            add_if_does_not_exist(entry, ("sample", "sulfateReagent", "instructionsEnUs"), "")
            add_if_does_not_exist(entry, ("sample", "sulfateReagent", "instructionsPtBr"), "")
        else:
            raise NotImplementedError
        add_if_does_not_exist(entry, ("sample", "sourceStock", "alkalinity"), None)
        add_if_does_not_exist(entry, ("sample", "sourceStock", "chlorideConcentration"), None)
        add_if_does_not_exist(entry, ("sample", "sourceStock", "phosphateConcentration"), None)
        add_if_does_not_exist(entry, ("sample", "sourceStock", "sulfateConcentration"), None)
        add_if_does_not_exist(entry, ("sample", "sourceStock", "instructionsEnUs"), "")
        add_if_does_not_exist(entry, ("sample", "sourceStock", "instructionsPtBr"), "")
        add_if_does_not_exist(entry, ("sample", "sourceStock", "alkalinityUnit"), "MILLIGRAM_PER_LITER_OF_BICARBONATE")
        add_if_does_not_exist(entry, ("sample", "sourceStock", "chlorideConcentrationUnit"), "MILLIGRAM_PER_LITER_OF_SODIUM_CHLORIDE")
        add_if_does_not_exist(entry, ("sample", "sourceStock", "phosphateConcentrationUnit"), "MILLIGRAM_PER_LITER_OF_PHOSPHATE")
        add_if_does_not_exist(entry, ("sample", "sourceStock", "sulfateConcentrationUnit"), "MILLIGRAM_PER_LITER_OF_SULFATE")
        add_if_does_not_exist(entry, ("sample", "stockFactor"), 1.0)
        # Update file.
        with open(path, "w", encoding="utf8") as file:
            file.write(json.dumps(entry, ensure_ascii=False))


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--base_dirs", nargs="+", default=[])
    args = parser.parse_args()
    if len(args.base_dirs) == 0:
        args.base_dirs = DEFAULT_BASE_DIRS
    main(args)
