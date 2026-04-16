import os, json
from typing import List, Dict, Tuple, Any
import shutil
from tqdm import tqdm

ANALYTE = "suspension"
OVERRIDE = False

# PATHS must contain only jsons from type 2
PATHS = ["y:\\repos\\Dataset-Injecao\\train"]

ANALYTES = {
        "alkalinity": "Alkalinity",
        "bisulfite": "Bisulfite",
        "emulsion": "Emulsion",
        "iron2" : "Iron2",
        "iron3" : "Iron3",
        "ph" : "Ph",
        "redox" : "Redox",
        "sulfate" : "Sulfate",
        "suspended" : "Suspended",
        "suspension": "IronOxide",
        "totaliron" : "TotalIron",
        }

def all_files_and_dirs(base_dirs: List[str]) -> Tuple[Dict[str, Any], Dict[str, str], List[str]]:
    jsons: Dict[str, Any] = dict()
    jpgs: Dict[str, str] = dict()
    dirs: List[str] = list()
    not_visited_dirs = list(base_dirs)

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

    return jsons, jpgs, dirs

def fix_long_path(path):
    if os.name == 'nt' and not path.startswith('\\\\?\\'):
        if path.startswith('\\\\'):
            return '\\\\?\\UNC\\' + path[2:]
        else:
            return '\\\\?\\' + os.path.abspath(path)
    return path

def convert_sample(analyte: str, path: List[str], analytes_names: Dict[str, str], override: bool = False) -> None:
    '''converts json from type 1 (app 1.x) to type 2 (app 2.x)'''
    ANALYTES = analytes_names

    analyte = analyte.lower().replace(" ","")

    if analyte not in ANALYTES.keys():
        raise NotImplementedError

    #support internal function
    def analyte_concentration(source_t2):
        stock_name_json_t2 = concentration_json_t2 = source_t2["aliquot"]["stock"]["name"]
        analyte_json_t2 = source_t2["aliquot"]["stock"]["values"][0]["analyticalParameterKey"]["analyte"].lower()
        concentration_json_t2 = source_t2["aliquot"]["stock"]["values"][0]["value"]
        assert analyte == analyte_json_t2, f"Analytical Parameters aren`t the same: {analyte}, {analyte_json_t2}"

        SourceStock_json_t1 = {"sourceStock": {
                                        "name": stock_name_json_t2,
                                        "alkalinity": None,
                                        "chlorideConcentration": None,
                                        "phosphateConcentration": None,
                                        "sulfateConcentration": None,
                                        "iron2Concentration": None,
                                        "iron3Concentration": None,
                                        "bisulfiteConcentration": None,
                                        "emulsionConcentration": None,
                                        "suspendedConcentration": None,
                                        "ironOxideConcentration": None,
                                        "pHValue": None,
                                        "redoxValue": None,
                                        "mdtSolids": None,
                                        "components": [],
                                        "aliquots": [],
                                        "instructionsEnUs": "",
                                        "instructionsPtBr": "",
                                        "alkalinityUnit": "MILLIGRAM_PER_LITER_OF_BICARBONATE",
                                        "chlorideConcentrationUnit": "MILLIGRAM_PER_LITER_OF_SODIUM_CHLORIDE",
                                        "phosphateConcentrationUnit": "MILLIGRAM_PER_LITER_OF_PHOSPHATE",
                                        "sulfateConcentrationUnit": "MILLIGRAM_PER_LITER_OF_SULFATE",
                                        "iron3ConcentrationUnit": "MILLIGRAM_PER_LITER_OF_IRON3",
                                        "bisulfiteConcentrationUnit": "MILLIGRAM_PER_LITER_OF_BISULFITE",
                                        "iron2ConcentrationUnit": "MILLIGRAM_PER_LITER_OF_IRON2",
                                        "emulsionConcentrationUnit": "PARTS_PER_MILLION",
                                        "suspendedConcentrationUnit": "PARTS_PER_MILLION",
                                        "ironOxideConcentrationUnit": "MILLIGRAM",
                                        "pHValueUnit": "POWER_OF_HYDROGEN",
                                        "redoxValueUnit": "MILLIVOLTS",
                                        }
                                        }
        if analyte_json_t2 == "alkalinity":
            SourceStock_json_t1["sourceStock"]["alkalinity"] = concentration_json_t2
            return SourceStock_json_t1

        if analyte_json_t2 == "chloride":
            SourceStock_json_t1["sourceStock"]["chlorideConcentration"] = concentration_json_t2
            return SourceStock_json_t1

        if analyte_json_t2 == "phosphate":
            SourceStock_json_t1["sourceStock"]["phosphateConcentration"] = concentration_json_t2
            return SourceStock_json_t1

        if analyte_json_t2 == "sulfate":
            SourceStock_json_t1["sourceStock"]["sulfateConcentration"] = concentration_json_t2
            return SourceStock_json_t1

        if analyte_json_t2 == "bisulfite":
            SourceStock_json_t1["sourceStock"]["bisulfiteConcentration"] = concentration_json_t2
            return SourceStock_json_t1

        if analyte_json_t2 == "emulsion":
            SourceStock_json_t1["sourceStock"]["emulsionConcentration"] = concentration_json_t2
            return SourceStock_json_t1

        if analyte_json_t2 == "iron2":
            SourceStock_json_t1["sourceStock"]["iron2Concentration"] = concentration_json_t2
            return SourceStock_json_t1

        if analyte_json_t2 == "iron3":
            SourceStock_json_t1["sourceStock"]["iron3Concentration"] = concentration_json_t2
            return SourceStock_json_t1

        if analyte_json_t2 == "ph":
            SourceStock_json_t1["sourceStock"]["pHValue"] = concentration_json_t2
            return SourceStock_json_t1

        if analyte_json_t2 == "redox":
            SourceStock_json_t1["sourceStock"]["redoxValue"] = concentration_json_t2
            return SourceStock_json_t1

        if analyte_json_t2 == "suspended":
            SourceStock_json_t1["sourceStock"]["suspendedConcentration"] = concentration_json_t2
            return SourceStock_json_t1
        
        if analyte_json_t2 == "suspension":
            SourceStock_json_t1["sourceStock"]["ironOxideConcentration"] = concentration_json_t2
            return SourceStock_json_t1

        if analyte_json_t2 == "totaliron":
            SourceStock_json_t1["sourceStock"]["totalironConcentration"] = concentration_json_t2
            return SourceStock_json_t1

    #support internal function
    def analyte_reagent(source_t2, inputs_t2):
        analyte_json_t2 = source_t2["aliquot"]["stock"]["values"][0]["analyticalParameterKey"]["analyte"].lower()
        inputs_t2 = inputs_t2
        #stock_json_t2 = source_t2["aliquot"]["stock"]

        assert analyte == analyte_json_t2, f"Analytical Parameters aren`t the same: {analyte}, {analyte_json_t2}"

        components_json_t1 = {}
        if analyte_json_t2 == "alkalinity":
            reagents = { # maps the names from json type2 (key) to type1 (value)
                        "ColorReagent": "colorReagent",}

            if len(inputs_t2) != len(reagents.keys()):
                raise NameError(f"Number of reagents is higher than expected. Verify reagents in the sample json.")

            for component in inputs_t2:
                for i in range(len(inputs_t2[component])):
                    components_json_t1[reagents.get(component)] = {
                                                                "name": inputs_t2[component][i]["name"],
                                                                "components": [{"name": item["name"].get("ptBR", None),
                                                                            "concentration": item.get("concentration", None),
                                                                            "function": item.get("function", None),
                                                                            "batch": ""
                                                                            } for item in inputs_t2[component][i]["components"]]}
            return components_json_t1

        if analyte_json_t2 == "chloride":
            raise NotImplementedError(f'ver o nome usado no json T2 e colocar o reagente para {analyte_json_t2}')

        if analyte_json_t2 == "phosphate":
            raise NotImplementedError(f'ver o nome usado no json T2 e colocar o reagente para {analyte_json_t2}')

        if analyte_json_t2 == "sulfate":
            reagents = {"CB7": "sulfateReagent"} # maps the names from json type2 (key) to type1 (value)

            if len(inputs_t2["ReagentForSulfate"]) != len(reagents.keys()):
                    raise NameError(f"Number of reagents is higher than expected. Verify reagents in the sample json.")

            for i, reagent in enumerate(inputs_t2["ReagentForSulfate"]):
                name = reagent["name"]
                components_json_t1[reagents.get(name)] = {"name": inputs_t2["ReagentForSulfate"][i]["name"],
                                                                "components": [{"name": item["name"].get("ptBR", None),
                                                                            "concentration": item.get("concentration", None),
                                                                            "function": item.get("function", None),
                                                                            "batch": ""
                                                                            } for item in inputs_t2["ReagentForSulfate"][i]["components"]]}

            return components_json_t1

        if analyte_json_t2 == "bisulfite":
            reagents = { # maps the names from json type2 (key) to type1 (value)
                        "Reagente 1 (SR3)": "bisulfiteReagent",
                        "Reagente 2 (F0.1%)": "formaldehyde",
                        "Reagente 3 (Fv0)": "reagente3",
                        "Reagente 4 (S8)": "reagente4"}

            for component in inputs_t2:
                if len(inputs_t2[component]) != len(reagents.keys()):
                    raise NameError(f"Number of reagents is higher than expected. Verify reagents in the sample json.")
                else:
                    for i in range(len(inputs_t2[component])):
                        components_json_t1[reagents.get(inputs_t2[component][i]["name"])] = {
                                                                    "name": inputs_t2[component][i]["name"],
                                                                    "components": [{"name": item["name"].get("ptBR", None),
                                                                                "concentration": item.get("concentration", None),
                                                                                "function": item.get("function", None),
                                                                                "batch": ""
                                                                                } for item in inputs_t2[component][i]["components"]]}
            return components_json_t1

        if analyte_json_t2 == "emulsion":
            raise NotImplementedError(f'ver o nome usado no json T2 e colocar o reagente para {analyte_json_t2}')

        if analyte_json_t2 == "iron2":
            reagents = { # maps the names from json type2 (key) to type1 (value)
                        "ComplexantMix": "complexant",
                        "Buffer": "buffer"}

            if len(inputs_t2) != len(reagents.keys()):
                raise NameError(f"Number of reagents is higher than expected. Verify reagents in the sample json.")

            for component in inputs_t2:
                for i in range(len(inputs_t2[component])):
                    components_json_t1[reagents.get(component)] = {
                                                                "name": inputs_t2[component][i]["name"],
                                                                "components": [{"name": item["name"].get("ptBR", None),
                                                                            "concentration": item.get("concentration", None),
                                                                            "function": item.get("function", None),
                                                                            "batch": ""
                                                                            } for item in inputs_t2[component][i]["components"]]}
            return components_json_t1

        if analyte_json_t2 == "iron3":
            raise NotImplementedError(f'ver o nome usado no json T2 e colocar o reagente para {analyte_json_t2}')

        if analyte_json_t2 == "ph":
            raise NotImplementedError(f'ver o nome usado no json T2 e colocar o reagente para {analyte_json_t2}')

        if analyte_json_t2 == "redox":
            raise NotImplementedError(f'ver o nome usado no json T2 e colocar o reagente para {analyte_json_t2}')

        if analyte_json_t2 == "suspended":
            raise NotImplementedError(f'ver o nome usado no json T2 e colocar o reagente para {analyte_json_t2}')

        if analyte_json_t2 == "suspension":
            return {"INTERNAL_USE": {"name": "internal_use", "components": []}}

        if analyte_json_t2 == "totaliron":
            raise NotImplementedError(f'ver o nome usado no json T2 e colocar o reagente para {analyte_json_t2}')

    jsons, jpgs, _ = all_files_and_dirs(path)

    list_of_paths = []
    for name, value in jsons.items():
          list_of_paths.append(value[1])

    for file in tqdm(sorted(list_of_paths), desc = "processing samples"):
        if file.endswith('json') and os.path.basename(file) != "Data.json":
            if override:
                dir_name = os.path.dirname(file)
                file_name = os.path.basename(file)
                prefixed_name = f"{ANALYTE.capitalize()}TrainingSample_{file_name}"
                save_path = os.path.join(dir_name, prefixed_name)
            else:
                base = PATHS[0]
                rel_path = os.path.relpath(file, base)
                dir_name = os.path.dirname(rel_path)
                file_name = os.path.basename(rel_path)
                prefixed_name = f"{ANALYTE.capitalize()}TrainingSample_{file_name}"

                save_dir = os.path.join(base, "adapted-jsons-type1", dir_name)
                os.makedirs(save_dir, exist_ok=True)
                save_path = os.path.join(save_dir, prefixed_name)


            with open(f'{file}', "r", encoding="utf8") as file_open:
                json_t2 = json.load(file_open)
                json_t1 = {}

                #key_t2 = json_t2.pop("key")

                source_t2 = json_t2["readingMedium"].pop("source")
                inputs_t2 = json_t2["readingMedium"].pop("inputs")
                ChamberType_t2 = json_t2["readingMedium"].pop("chamberType").split("_")[0]
                flash_t2 = json_t2["readingMedium"].pop("flashlight")
                vibration_t2 = json_t2["readingMedium"].pop("vibration")

                StandardVolume_t2 = json_t2["extraValues"].get("StandardVolume", 1000.0)
                UsedVolume_t2 = json_t2["extraValues"].get("UsedVolume", 1000.0)

                filename_t2 = json_t2.pop("fileName")
                extrafilename_t2 = json_t2.pop("extraFileNames")
                blank_t2 = json_t2.pop("referenceFileName")
                analyst_t2 = json_t2.pop("analystName")
                notes_t2 = json_t2.pop("notes")
                datetime_t2 = json_t2.pop("datetime")
                VolumeUnit_t2 = source_t2["aliquot"]["finalVolumeUnit"]

                app_t2 = json_t2.pop("app")
                device_t2 = json_t2.pop("device")
                # try: setup_t2 = json_t2.pop("setup")
                # except KeyError: setup_t2 = None

                # builds a json type 1
                json_t1["app"] = {
                                    "packageName": app_t2["pkg"],
                                    "appName": app_t2["name"],
                                    "versionName": app_t2["version"],
                                    }
                json_t1["device"] = {
                                        "alias": device_t2["name"],
                                        "model": device_t2["model"],
                                        "manufacturer": device_t2["manufacturer"],
                                        "androidVersion": device_t2["androidVersion"],
                                        }
                json_t1["sample"] = {}
                reagents = analyte_reagent(source_t2, inputs_t2)
                for key in reagents.keys():
                    json_t1["sample"][key] = reagents.get(key)
                json_t1["sample"]["sourceStock"] = analyte_concentration(source_t2).get("sourceStock")
                json_t1["sample"]["sourceAliquot"] = {
                                            "name": source_t2["aliquot"]["name"],
                                            "finalVolume": source_t2["aliquot"]["finalVolume"],
                                            "aliquot": source_t2["aliquot"]["aliquot"],
                                            "finalVolumeUnit": source_t2["aliquot"]["finalVolumeUnit"],
                                            "aliquotUnit": source_t2["aliquot"]["aliquotUnit"],}
                json_t1["sample"]["stockFactor"] = source_t2["stockFactor"]
                json_t1["sample"]["standardVolume"] = StandardVolume_t2
                json_t1["sample"]["usedVolume"] = UsedVolume_t2
                json_t1["sample"]["fileName"] = f"{ANALYTE.capitalize()}TrainingSample_{filename_t2}"
                json_t1["sample"]["extraFileNames"] = [f"{ANALYTE.capitalize()}TrainingSample_{name}" for name in extrafilename_t2]
                json_t1["sample"]["blankFileName"] = f"{ANALYTE.capitalize()}TrainingSample_{blank_t2}" if blank_t2 != None else None
                json_t1["sample"]["analystName"] = analyst_t2
                json_t1["sample"]["notes"] = notes_t2
                json_t1["sample"]["vibration"] = vibration_t2
                json_t1["sample"]["flash"] = flash_t2
                json_t1["sample"]["datetime"] = datetime_t2
                json_t1["sample"]["chamberType"] = ChamberType_t2
                json_t1["sample"]["volumeUnit"] = VolumeUnit_t2

                # TODO verificar com o professor se é a melhor forma de lidar com os extraValues, visto que eles podem variar muito de um json para outro. Talvez seja interessante criar uma função específica para lidar com eles, ou até mesmo criar campos específicos no json_t1 para os extraValues mais comuns.
                # json_t1["sample"]["extraValues"] = json_t2.get("extraValues", {})
                json_t1["sample"]["extraValues"] = {k:json_t2["extraValues"][k] for k in list(json_t2.get("extraValues", {}).keys())}

            save_path = fix_long_path(save_path)
            os.makedirs(os.path.dirname(save_path), exist_ok=True)

            with open(save_path, "w", encoding="utf8") as f:
                json.dump(json_t1, f, indent=2, ensure_ascii=False)

            if override and file != save_path:
                os.remove(file)

            json_stem = os.path.splitext(os.path.basename(file))[0]
            matching_jpgs = [
                jpg_path for jpg_name, jpg_path in jpgs.items()
                if jpg_name.startswith(json_stem)
            ]

            for jpg_path in matching_jpgs:
                jpg_name = os.path.basename(jpg_path)
                new_jpg_name = f"{ANALYTE.capitalize()}TrainingSample_{jpg_name}"

                if override:
                    jpg_save_path = os.path.join(os.path.dirname(save_path), new_jpg_name)
                    shutil.move(jpg_path, jpg_save_path)
                else:
                    jpg_save_path = os.path.join(os.path.dirname(save_path), new_jpg_name)
                    os.makedirs(os.path.dirname(jpg_save_path), exist_ok=True)
                    shutil.copy2(jpg_path, jpg_save_path)

    if override:
        for base_path in path:
            for root, dirs, files in os.walk(base_path):
                for filename in files:
                    if filename == "Data.json":
                        data_json_path = os.path.join(root, filename)
                        os.remove(data_json_path)


if __name__ == "__main__":
    convert_sample(analyte= ANALYTE, path = PATHS, analytes_names=ANALYTES, override=OVERRIDE)