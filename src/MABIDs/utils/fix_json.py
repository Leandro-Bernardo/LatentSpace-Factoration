import os, json
from typing import List, Dict, Tuple, Any

OLD_ANALYTE_NAME = "Iron3"
TRUE_ANALYTE_NAME = "TotalIron"
#PATHS = [f"G:\\Dataset\\{TRUE_ANALYTE_NAME}\\MinIO\\validation"]#\\Validacao\\{TRUE_ANALYTE_NAME}TrainingSamples"]

PATHS = [f"G:\\DAVID\\{TRUE_ANALYTE_NAME}\\CellphoneSamples\\Iron3"]#\\Validacao\\{TRUE_ANALYTE_NAME}TrainingSamples"] G:\DAVID\TotalIron\TrainingSamples\Ferro Total

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
        "totaliron" : "TotalIron",
        }

def all_files_and_dirs(base_dirs: List[str]) -> Tuple[Dict[str, Any], Dict[str, str], List[str]]:
    jsons: Dict[str, Any] = dict()  # {json_filename: [sample, path]}
    jpgs: Dict[str, str] = dict()   # {jpg_filename: path}
    dirs: List[str] = list()        # [dir, ...]
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

#renames all files found from (old name) to (new name)
def rename_file(old_analyte_name: str, analyte: str, path: List[str], analytes_names: Dict[str,str]) -> None:

    ANALYTES = analytes_names#list(analytes_names.keys())

    old_analyte_name = old_analyte_name.capitalize().replace(" ","")
    analyte = ANALYTES.get(analyte.lower().replace(" ",""), None)

    _,_,dirs = all_files_and_dirs(path)

    for path in dirs:
        for file in sorted(os.listdir(path)):
            if old_analyte_name in file:

                specific_name = file.split(f'{old_analyte_name}')
                os.rename(f'{path}/{file}',f'{path}/{analyte}{specific_name[1]}')

            else: pass

#fixes the json from each file found, that were created as EXP alkalinity, based on what the given real analyte should be
#TODO if needed, add to old analytes (alkalinity,chloride, phosphate, sulfate) the concentration unity of newerest analytes
def fix_sample(old_analyte_name: str, analyte: str, path: List[str], analytes_names: Dict[str, str]) -> None:

    ANALYTES = analytes_names

    old_analyte_name = old_analyte_name.lower().replace(" ","")
    analyte = analyte.lower().replace(" ","")

    if analyte not in ANALYTES.keys():
        raise NotImplementedError

    #support internal function
    def get_analyte_concentration(old_analyte_name):

        if old_analyte_name == "alkalinity":
            return old_sample["sourceStock"]["alkalinity"]

        if old_analyte_name == "chloride":
            return old_sample["sourceStock"]["chlorideConcentration"]

        if old_analyte_name == "phosphate":
            return old_sample["sourceStock"]["phosphateConcentration"]

        if old_analyte_name == "sulfate":
            return old_sample["sourceStock"]["sulfateConcentration"]

        if old_analyte_name == "bisulfite":
            return old_sample["sourceStock"]["bisulfiteConcentration"]

        if old_analyte_name == "emulsion":
            return old_sample["sourceStock"]["emulsionConcentration"]

        if old_analyte_name == "iron2":
            return old_sample["sourceStock"]["iron2Concentration"]

        if old_analyte_name == "iron3":
            return old_sample["sourceStock"]["iron3Concentration"]

        if old_analyte_name == "ph":
            return old_sample["sourceStock"]["pHValue"]

        if old_analyte_name == "redox":
            return old_sample["sourceStock"]["redoxValue"]

        if old_analyte_name == "suspended":
            return old_sample["sourceStock"]["suspendedConcentration"]

        if old_analyte_name == "totaliron":
            return old_sample["sourceStock"]["totalironConcentration"]

    #support internal function
    def true_ph_value(old_sample, old_analyte_name):  #if file is a blank returns 7 (neutral ph), if file is a sample, return sample value
        if "Branco" in old_sample["sourceAliquot"]["name"] or "Zeragem" in old_sample["sourceAliquot"]["name"] :
            return 7.0

        elif "pH" in old_sample["sourceAliquot"]["name"]:
            return old_sample["sourceStock"]["pHValue"]
        else:
            return old_sample["sourceStock"]["alkalinity"]

    #support internal function
    def get_concentration_unit(analyte):
        if analyte == "alkalinity":
            return "MILLIGRAM_PER_LITER_OF_BICARBONATE"

        if analyte == "chloride":
            return "MILLIGRAM_PER_LITER_OF_SODIUM_CHLORIDE"

        if analyte == "phosphate":
            return  "MILLIGRAM_PER_LITER_OF_PHOSPHATE"

        if analyte == "sulfate":
            return "MILLIGRAM_PER_LITER_OF_SULFATE"

        if analyte == "bisulfite":
            return "MILLIGRAM_PER_LITER_OF_BISULFITE"

        if analyte == "emulsion":
            return "PARTS_PER_MILLION"

        if analyte == "iron2":
            return "MILLIGRAM_PER_LITER_OF_IRON2"

        if analyte == "iron3":
            return "MILLIGRAM_PER_LITER_OF_IRON3"

        if analyte == "ph":
            return "POWER_OF_HYDROGEN"

        if analyte == "redox":
            return "MILLIVOLTS"

        if analyte == "suspended":
            return "PARTS_PER_MILLION"

        if analyte == "totaliron":
            return "MILLIGRAM_PER_LITER_OF_TOTAL_IRON"


    jsons, _, _ = all_files_and_dirs(path)

    list_of_paths = []
    for name, value in jsons.items():
          list_of_paths.append(value[1])

    for file in sorted(list_of_paths):
        if file.endswith('json'):
            with open(f'{file}', "r", encoding="utf8") as file_open:
                file_json = json.load(file_open)
                #the JSONS have common tags (like 'app', 'device', etc) that are independent of the analytes. The following lines takes those common tags
                old_app = file_json.pop("app")
                old_device = file_json.pop("device")
                try: old_setup = file_json.pop("setup")
                except KeyError: old_setup = None
                old_sample = file_json.pop("sample")

                file_json["app"] = {
                                    "packageName": old_app["packageName"] if old_app.get("packageName") else old_app["package_name"],
                                    "appName": old_app["appName"] if old_app.get("appName") else old_app["app_name"],
                                    "versionName": old_app["versionName"] if old_app.get("versionName") else old_app["version_name"],
                                    }

                file_json["device"] = {
                                        "model": old_device["model"],
                                        "manufacturer": old_device["manufacturer"],
                                        "androidVersion": old_device["androidVersion"] if old_device.get("androidVersion") else old_device["android_version"]
                                        }
                file_json["setup"] = old_setup

            #takes the unique tags for every analyte (if exists) and add it to the fixed JSON
            if analyte == "alkalinity":
                first_part_of_sample = {
                                        "colorReagent": {
                                                        "name": "CR11",
                                                        "components": [
                                                            {
                                                            "name": "Azul de Bromofenol",
                                                            "concentration": 500.0,
                                                            "batch": "",
                                                            "function": "DYE"
                                                            },
                                                            {
                                                            "name": "Cloreto de Sódio",
                                                            "concentration": 2.0,
                                                            "batch": "",
                                                            "function": "ION"
                                                            },
                                                            {
                                                            "name": "Etanol",
                                                            "concentration": 15.0,
                                                            "batch": "",
                                                            "function": "COSOLVENT"
                                                            },
                                                            {
                                                            "name": "Ácido Fórmico",
                                                            "concentration": 0.0045,
                                                            "batch": "",
                                                            "function": "ACID"
                                                            }
                                                        ],
                                                        "instructionsEnUs": "<b>I1</b>\nPrepare 500 mg/L of bromophenol blue solution in ethanol P.A. (suggested volume = 250 mL).\n\n<b>F0</b>\nPrepare and standardize 0.1 mol/L formic acid solution (suggested volume = 250 mL).\n\n<b>S10</b>\nPrepare a 2.5 mol/L sodium chloride solution (suggested volume per calibration series = 250 mL).\n\n<b>CR11</b>\nIn a 250 mL volumetric flask, add a volume of F0 such that the final concentration of formic acid in CR11 solution should be 0.0045 mol/L (use the F0 concentration determined in standardization to calculate this volume), 25 mL of I1, 12 mL of ethanol P.A. and complete the volume with S10. Maintain refrigerated.\n",
                                                        "instructionsPtBr": "<b>I1</b>\nPreparar solução de azul de bromofenol 500 mg/L em etanol P.A. (volume sugerido = 250 mL).\n\n<b>F0</b>\nPreparar e padronizar solução de ácido fórmico 0,1 mol/L (volume sugerido = 250 mL).\n\n<b>S10</b>\nPreparar solução de cloreto de sódio 2,5 mol/L (volume sugerido por série de calibração = 250 mL).\n\n<b>CR11</b>\nEm balão volumétrico de 250 mL, adicionar F0 para que a concentração final no reagente colorido (CR11) seja 0,0045 mol/L de ácido fórmico (usar a concentração de F0 encontrada na padronização para o cálculo deste volume), 25 mL de I1, 12 mL de etanol P.A. e avolumar com S10. Manter sob refrigeração."
                                                            },
                                        }

            if analyte == "chloride":
                first_part_of_sample = {
                                        "chlorideReagent": {
                                                            "name": "NP1",
                                                            "components": [
                                                                {
                                                                "name": "Nitrato de Prata",
                                                                "concentration": 0.6,
                                                                "batch": "",
                                                                "function": "PRECIPITANT"
                                                                },
                                                                {
                                                                "name": "Ácido Nítrico",
                                                                "concentration": 0.5,
                                                                "batch": "",
                                                                "function": "ACID"
                                                                }
                                                            ],
                                                            "instructionsEnUs": "<b>HNO3-0,5M</b>\nPrepare a nitric acid solution (suggested volume per calibration series = 50 mL).\n\n<b>NP1</b>\nPrepare a 0.6 mol/L silver nitrate solution in HNO3-0.5M (suggested volume per calibration series = 25 mL).",
                                                            "instructionsPtBr": "<b>HNO3-0,5M</b>\nPreparar solução de ácido nítrico 0,5 mol/L (volume sugerido por série de calibração = 50 mL).\n\n<b>NP1</b>\nPreparar solução de nitrato de prata 0,6 mol/L em HNO3-0,5M (volume sugerido por série de calibração = 25 mL)."
                                                            },
                                            "liquidator": {
                                                        "name": "FP1",
                                                        "components": [
                                                            {
                                                            "name": "Dihidrogenofosfato de Potássio",
                                                            "concentration": 1.0,
                                                            "batch": "",
                                                            "function": "LIQUIDATOR"
                                                            }
                                                        ],
                                                        "instructionsEnUs": "Prepare an 1 mol/L monobasic potassium phosphate solution in deionized water (suggested volume per calibration series = 25 mL).",
                                                        "instructionsPtBr": "Preparar solução de fosfato de potássio monobásico 1 mol/L em água deionizada (volume sugerido por série de calibração = 25 mL)."
                                                        },
                                        }

            if analyte == "bisulfite":
                first_part_of_sample = {}

            if analyte == "emulsion":
                first_part_of_sample = {}

            if analyte == "iron2":
                first_part_of_sample = {
                                        "complexant":{
                                                    "name": "O- Fenantrolina 0.005  M",
                                                    "components": [],
                                                    "instructionsEnUs": "",
                                                    "instructionsPtBr": ""
                                                        },
                                        "buffer":{
                                                "name": "Ácido Ácetico-Acetato - PH 5",
                                                "components": [],
                                                "instructionsEnUs": "",
                                                "instructionsPtBr": ""
                                                    },
                                        }

            if analyte == "iron3":
                first_part_of_sample = {
                                        "complexant" : {
                                                        "name": "KSCN 90% de saturação",
                                                        "components": [],
                                                        "instructionsEnUs": "",
                                                        "instructionsPtBr": ""
                                                        },
                                        "acid" : {
                                                "name": "HCl 4M",
                                                "components": [],
                                                "instructionsEnUs": "",
                                                "instructionsPtBr": ""
                                                },
                                        }

            if analyte == "ph":
                first_part_of_sample = {
                                        "phIndicatorMix": {
                                                        "name": "IND6",
                                                        "components": [],
                                                        "instructionsEnUs": "",
                                                        "instructionsPtBr": ""
                                                        },
                                        }

            if analyte == "redox":
                first_part_of_sample = {}

            if analyte == "sulfate":
                first_part_of_sample = {}

            if analyte == "suspended":
                first_part_of_sample = {}

            if analyte == "totaliron":
                first_part_of_sample = {
                                        "complexant": {
                                                    "name": "KSCN",
                                                    "components": [
                                                        {
                                                        "name": "KSCN - 80% do limite de saturação",
                                                        "concentration": 750.0,
                                                        "batch": "",
                                                        "function": "COMPLEXING"
                                                        }
                                                    ],
                                                    "instructionsEnUs": "",
                                                    "instructionsPtBr": ""
                                                    },
                                        "acid": {
                                                "name": "S11",
                                                "components": [
                                                    {
                                                    "name": "Cloreto de Sódio",
                                                    "concentration": 1.0,
                                                    "batch": "",
                                                    "function": "ION"
                                                    },
                                                    {
                                                    "name": "Ácido Clorídrico",
                                                    "concentration": 4.0,
                                                    "batch": "",
                                                    "function": "ACID"
                                                    }
                                                ],
                                                "instructionsEnUs": "",
                                                "instructionsPtBr": ""
                                                },
                                            }


            second_part_of_sample = {
                                    "sourceStock": {
                                                    "name": old_sample["sourceStock"]["name"].replace("EXP","").replace("exp",""),
                                                    "alkalinity": get_analyte_concentration(old_analyte_name) if analyte == "alkalinity" else None,
                                                    "chlorideConcentration": get_analyte_concentration(old_analyte_name) if analyte == "chloride" else None,
                                                    "phosphateConcentration": get_analyte_concentration(old_analyte_name) if analyte == "phosphate" else None,
                                                    "sulfateConcentration": get_analyte_concentration(old_analyte_name) if analyte == "sulfate" else None,
                                                    "iron2Concentration": get_analyte_concentration(old_analyte_name) if analyte == "iron2" else None,
                                                    "iron3Concentration": get_analyte_concentration(old_analyte_name) if analyte == "iron3" else None ,
                                                    "bisulfiteConcentration": get_analyte_concentration(old_analyte_name) if analyte == "bisulfite" else None,
                                                    "emulsionConcentration": get_analyte_concentration(old_analyte_name) if analyte == "emulsion" else None,
                                                    "suspendedConcentration": get_analyte_concentration(old_analyte_name) if analyte == "suspended" else None,
                                                    "pHValue": true_ph_value(old_sample, old_analyte_name) if analyte == "ph" else None,
                                                    "redoxValue": get_analyte_concentration(old_analyte_name) if analyte == "redox" else None,
                                                    "totalironConcentration": get_analyte_concentration(old_analyte_name) if analyte == "totaliron" else None,

                                                    "components": [], #old_sample["sourceStock"]["components"],
                                                    "aliquots": [], #old_sample["sourceStock"]["aliquots"],
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
                                                    "pHValueUnit": "POWER_OF_HYDROGEN",
                                                    "redoxValueUnit": "MILLIVOLTS",
                                                    "totalironConcentrationUnit" : "MILLIGRAM_PER_LITER_OF_TOTAL_IRON",
                                                    },

                                    "sourceAliquot" : {
                                                        "name": old_sample["sourceAliquot"]["name"],
                                                        "finalVolume": old_sample["sourceAliquot"]["finalVolume"],
                                                        "aliquot": old_sample["sourceAliquot"]["aliquot"],
                                                        "finalVolumeUnit": old_sample["sourceAliquot"]["finalVolumeUnit"] if old_sample["sourceAliquot"].get("finalVolumeUnit") else old_sample["sourceAliquot"]["finalVolumeUnity"],
                                                        "aliquotUnit": old_sample["sourceAliquot"]["aliquotUnit"]  if old_sample["sourceAliquot"].get("aliquotUnit") else old_sample["sourceAliquot"]["aliquotUnity"]
                                                        },
                                    "stockFactor": old_sample["stockFactor"],
                                    "standardVolume": old_sample["standardVolume"],
                                    "usedVolume": old_sample["usedVolume"],
                                    "fileName": old_sample["fileName"].replace(old_analyte_name.capitalize(), ANALYTES.get(analyte.lower().replace(" ",""), None)),
                                    "extraFileNames": [ extraFile.replace(old_analyte_name.capitalize(), ANALYTES.get(analyte.lower().replace(" ",""), None)) for extraFile in old_sample["extraFileNames"] ],
                                    "blankFileName": old_sample["blankFileName"].replace(old_analyte_name.capitalize(), ANALYTES.get(analyte.lower().replace(" ",""), None)) if old_sample.get("blankFileName") else None,
                                    "analystName": old_sample["analystName"],
                                    "notes": old_sample["notes"],
                                    "datetime": old_sample["datetime"],
                                    "chamberType": old_sample["chamberType"],
                                    "concentrationUnit": get_concentration_unit(analyte),
                                    "volumeUnit": old_sample["volumeUnit"] if old_sample.get("volumeUnit") else old_sample["volumeUnity"]
                                }
            file_json["sample"] = {**first_part_of_sample, **second_part_of_sample}

            with open(f'{file}', "w", encoding="utf8") as file_save:
                file_save.write(json.dumps(file_json, ensure_ascii=False))


def main():

    rename_file(old_analyte_name = OLD_ANALYTE_NAME, analyte = TRUE_ANALYTE_NAME, path = PATHS, analytes_names=ANALYTES)

    fix_sample(old_analyte_name = OLD_ANALYTE_NAME, analyte= TRUE_ANALYTE_NAME, path = PATHS, analytes_names=ANALYTES)


main()



