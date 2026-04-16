import pandas as pd
import os
import json
import numpy as np

from tqdm import tqdm
from typing import Any, Dict, List, Tuple
from datetime import datetime

ANALYTE = "Suspended"  # "Emulsion"
#FILES_ROOT = [f"G:\\Dataset\\{ANALYTE}\\Validacao\\{ANALYTE}TrainingSamples"]
FILES_ROOT = [f"G:\\Dataset\\{ANALYTE}\\MinIO\\validation"]

def all_files_and_dirs(base_dirs: List[str]) -> Tuple[Dict[str, Any], Dict[str, str], List[str]]:
    excels: list[str] = list() #  [excel]
    jpgs: Dict[str, str] = dict()   # {jpg_filename: base dir path}
    dirs: List[str] = list()        # [dir, ...]
    not_visited_dirs = list(base_dirs)
    with tqdm(desc="Scanning folders", total=len(base_dirs)) as pbar:
        while len(not_visited_dirs) != 0:
            current_dir = not_visited_dirs.pop(0)
            dirs.append(current_dir)
            for filename in sorted(os.listdir(current_dir)):
                path = os.path.join(current_dir, filename)
                if os.path.isfile(path):
                    if filename.lower().endswith(".xlsx") or filename.lower().endswith(".xls"):
                            excels.append(path)
                    elif filename.lower().endswith(".jpg"):
                        jpgs[filename] = current_dir
                elif os.path.isdir(path):
                    not_visited_dirs.append(path)
                    pbar.total += 1
                    pbar.refresh()
            pbar.update(1)
    return excels, jpgs

def convert_to_serializable(obj):
    if isinstance(obj, (np.int64, np.float64)):
        return obj.item()  # Converte para tipo nativo (int ou float)
    if isinstance(obj, (np.ndarray,)):  # Para arrays do numpy
        return obj.tolist()  # Converte para lista
    raise TypeError(f"Type {type(obj)} not serializable")

excels, jpgs = all_files_and_dirs(FILES_ROOT)

all_months = {
    'jan.': '01',
    'fev.': '02',
    'mar.': '03',
    'abr.': '04',
    'mai.': '05',
    'jun.': '06',
    'jul.': '07',
    'ago.': '08',
    'set.': '09',
    'out.': '10',
    'nov.': '11',
    'dez.': '12'
}
def fix_datetime(date_time):
    date_time_date, date_time_hour = date_time.split(", ")[1].split(" - ")[0], date_time.split(", ")[1].split(" - ")[1]
    for month in all_months.keys():
        # fixes month format
        if month in date_time_date: # from str to num
            date_time_date = date_time_date.replace(month, all_months[month])
    date_time_date = datetime.strptime(date_time_date, "%d/%m/%Y").strftime("%Y.%m.%d") # separator (from / to .)

    date_time_fixed = f"{date_time_date} {date_time_hour}:00 -0300"

    return date_time_fixed

def analyte_concentration(analyte, sample_file):
    if analyte == "Suspended":
        concentration = int(df.loc[sample_file,"Notes"].split(" ppm")[0]) if "ppm" in df.loc[sample_file,"Notes"] else None
    elif analyte == "Emulsion":
        concentration = int(df.loc[sample_file, "Alkalinity (mg HCO3-/L)"]) if int(df.loc[sample_file, "Alkalinity (mg HCO3-/L)"]) != 0 else None
    return concentration

def aloquit_name(analyte, sample_file):
    if analyte == "Suspended":
        name = str(df.loc[sample_file,"Notes"].split(" ppm")[0]) + "ppm" if "ppm" in df.loc[sample_file,"Notes"] else "Solução de Zeragem (Branco)"
    elif analyte == "Emulsion":
        name = str(df.loc[sample_file, "Alkalinity (mg HCO3-/L)"]) + "ppm" if int(df.loc[sample_file, "Alkalinity (mg HCO3-/L)"]) != 0 else "Solução de Zeragem (Branco)"
    return name

def sourceStock_name(analyte, sample_file):
    if analyte == "Suspended":
        name = "EXP Suspended" if "ppm" in df.loc[sample_file,"Notes"] else "Uso Interno"
    elif analyte == "Emulsion":
        name = "EXP Emulsion" if int(df.loc[sample_file, "Alkalinity (mg HCO3-/L)"]) != 0 else "Uso Interno"
    return name


for excel in excels:
    # open the report excel
    df = pd.read_excel(excel)
    df.fillna(" ", inplace=True)
    df.set_index("Sample File", inplace=True)
    # iterates over each file name, recreating its json
    for sample_file in df.index:
        # creates a dictionary which will receive informations (some default and some from the excel)
        json_sample = {}

        json_sample["app"] = {"packageName": "com.prograf.chemicalanalysis",
                        "appName": "Análises Químicas",
                        "versionName": "1.3.2"}

        json_sample["device"] = {"model": "SM-A725M",
                        "manufacturer": "samsung",
                        "androidVersion": "SDK 30 (11)"}

        json_sample["setup"] = None

        json_sample["sample"] = {}
        json_sample["sample"]["colorReagent"] = {"name": "CR11",
                                            "components": [
                                                {
                                                "name": "Azul de Bromofenol",
                                                "concentration": 500.0,
                                                "batch": "243144 synth",
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
                                                "batch": "Etanol da Capela",
                                                "function": "COSOLVENT"
                                                },
                                                {
                                                "name": "Ácido Fórmico",
                                                "concentration": 0.0045,
                                                "batch": "",
                                                "function": "ACID"
                                                }
                                            ],
                                            "instructionsEnUs": "",
                                            "instructionsPtBr": ""}
        json_sample["sample"]["standardVolume"] = 1.0 #df.loc[sample_file, "Standard Volume (muL)"]
        json_sample["sample"]["usedVolume"] = 1.0 #df.loc[sample_file, "Used Volume (muL)"]
        json_sample["sample"]["chamberType"] = "POT"
        json_sample["sample"]["fileName"] = sample_file
        json_sample["sample"]["extraFileNames"] = []
        json_sample["sample"]["blankFileName"] = df.loc[sample_file,"Blank File"] if "ppm" in df.loc[sample_file,"Notes"] else None
        json_sample["sample"]["notes"] = df.loc[sample_file,"Notes"]
        json_sample["sample"]["analystName"] = df.loc[sample_file, "Analyst"]

        date_time = fix_datetime(df.loc[sample_file, "Date"])
        json_sample["sample"]["datetime"] = date_time

        json_sample["sample"]["sourceStock"] = {"name": sourceStock_name(ANALYTE, sample_file),
                                        "components": [{"name": "Acetato de Sódio",
                                                        "concentration": 0.1306,
                                                        "batch": "",
                                                        "concentrationUnit": "MOL_PER_LITER"},

                                                        { "name": "Cloreto de Sódio",
                                                        "concentration": 5.0,
                                                        "batch": "",
                                                        "concentrationUnit": "MOL_PER_LITER"}],

                                        "aliquots": [],
                                        "alkalinity": analyte_concentration(ANALYTE, sample_file),#df.loc[sample_file, "Alkalinity (mg HCO3-/L)"],
                                        "chlorideConcentration": None,
                                        "phosphateConcentration": None,
                                        "sulfateConcentration": None,
                                        "instructionsEnUs": "",
                                        "instructionsPtBr": "",
                                        "alkalinityUnit": "MILLIGRAM_PER_LITER_OF_BICARBONATE",
                                        "chlorideConcentrationUnit": "MILLIGRAM_PER_LITER_OF_SODIUM_CHLORIDE",
                                        "phosphateConcentrationUnit": "MILLIGRAM_PER_LITER_OF_PHOSPHATE",
                                        "sulfateConcentrationUnit": "MILLIGRAM_PER_LITER_OF_SULFATE"
                                        }

        json_sample["sample"]["sourceAliquot"] = {"name": f"EXP {ANALYTE}",
                                           "finalVolume": 1.0,
                                           "aliquot": 1.0,
                                           "aliquotUnit": "MILLILITER",
                                           "finalVolumeUnit": "MILLILITER"}

        json_sample["sample"]["stockFactor"] = 1.0 #df.loc[sample_file, "Stock Factor"] # TODO fix this later
        json_sample["sample"]["volumeUnit"] = "MICROLITER"

        save_path = os.path.join(jpgs[sample_file], f"{sample_file.strip(".jpg")}")

        with open(f"{save_path}.json", "w+", encoding="utf-8") as file:
            json.dump(json_sample, file, default=convert_to_serializable, ensure_ascii=False)
