import os
import json
import shutil
import numpy as np

from tqdm import tqdm
from typing import  Tuple, List, Dict, Any
import sys

# necessary to import chemical_analysis
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import chemical_analysis as ca

#variables
ANALYTE: str = "Alkalinity"
min_value: float = 0.0
max_value: float = 4000.0
PATH: List[str] = [f"G:\\Dataset\\{ANALYTE}\\this"]
SAVE_PATH: str = os.path.join(os.path.dirname(__file__), "..", "..", "..", "Dataset", f"{ANALYTE}", f"samples_in_range({min_value}, {max_value})")
CACHE_BASE_DIR: str = os.path.join(os.path.dirname(__file__), "..", "..", "..", "Dataset", "cache")
CACHE_PATH: str = os.path.join(CACHE_BASE_DIR, f"{ANALYTE}" , f"filter_by_range")

os.makedirs(SAVE_PATH, exist_ok = True)

ANALYTE = ANALYTE.lower()

dataset = {
            "alkalinity":{"dataset": ca.alkalinity.AlkalinitySampleDataset, "processed_dataset": ca.alkalinity.ProcessedAlkalinitySampleDataset},
            "chloride": {"dataset": ca.chloride.ChlorideSampleDataset, "processed_dataset": ca.chloride.ProcessedChlorideSampleDataset},
            "sulfate": {"dataset": ca.sulfate.SulfateSampleDataset, "processed_dataset": ca.sulfate.ProcessedSulfateSampleDataset},
            "phosphate": {"dataset": ca.phosphate.PhosphateSampleDataset, "processed_dataset": ca.phosphate.ProcessedPhosphateSampleDataset},
            "bisulfite": {"dataset": ca.bisulfite2d.Bisulfite2DSampleDataset, "processed_dataset": ca.bisulfite2d.ProcessedBisulfite2DSampleDataset},
            "iron2": {"dataset": ca.iron2.Iron2SampleDataset, "processed_dataset": ca.iron2.ProcessedIron2SampleDataset},
            "iron3": {"dataset": ca.iron3.Iron3SampleDataset, "processed_dataset": ca.iron3.ProcessedIron3SampleDataset},
            #"iron_oxid": {"dataset": ca.iron_oxid.IronOxidSampleDataset, "processed_dataset": ca.iron_oxid.ProcessedIronOxidSampleDataset},
            "ph": {"dataset": ca.ph.PhSampleDataset, "processed_dataset": ca.ph.ProcessedPhSampleDataset},
            }

pca_stats = {
            "bisulfite"  : {"lab_mean": np.load(ca.bisulfite2d.PCA_STATS)['lab_mean']  , "lab_sorted_eigenvectors": np.load(ca.bisulfite2d.PCA_STATS)['lab_sorted_eigenvectors']},
            "chloride"  : {"lab_mean": np.load(ca.chloride.PCA_STATS)['lab_mean']  , "lab_sorted_eigenvectors": np.load(ca.chloride.PCA_STATS)['lab_sorted_eigenvectors']},
            "iron2"  : {"lab_mean": np.load(ca.iron2.PCA_STATS)['lab_mean']  , "lab_sorted_eigenvectors": np.load(ca.iron2.PCA_STATS)['lab_sorted_eigenvectors']},
            "iron3"  : {"lab_mean": np.load(ca.iron3.PCA_STATS)['lab_mean']  , "lab_sorted_eigenvectors": np.load(ca.iron3.PCA_STATS)['lab_sorted_eigenvectors']},
            #"iron_oxid"  : {"lab_mean": np.load(ca.iron_oxid.PCA_STATS)['lab_mean']  , "lab_sorted_eigenvectors": np.load(ca.iron_oxid.PCA_STATS)['lab_sorted_eigenvectors']},
            "phosphate"  : {"lab_mean": np.load(ca.phosphate.PCA_STATS)['lab_mean']  , "lab_sorted_eigenvectors": np.load(ca.phosphate.PCA_STATS)['lab_sorted_eigenvectors']},
            }
SampleDataset = dataset[f"{ANALYTE.lower()}"]["dataset"]
ProcessedSampleDataset = dataset[f"{ANALYTE.lower()}"]["processed_dataset"]

#data preprocessing
samples = SampleDataset(
    base_dirs = PATH,
    progress_bar = True,
    skip_blank_samples = True,
    skip_incomplete_samples = True,
    skip_inference_sample= True,
    skip_training_sample = False,
    verbose = True
)
if ANALYTE in pca_stats.keys(): # does have PCA
    processed_samples = ProcessedSampleDataset(
            dataset = samples,
            cache_dir = CACHE_PATH,
            num_augmented_samples = 0,
            progress_bar = True,
            transform = None,
            lab_mean= pca_stats[f"{ANALYTE}"]['lab_mean'],
            lab_sorted_eigenvectors = pca_stats[f"{ANALYTE}"]['lab_sorted_eigenvectors'])
else: # doenst have PCA
    processed_samples = ProcessedSampleDataset(
        dataset = samples,
        cache_dir = CACHE_PATH,
        num_augmented_samples = 0,
        progress_bar = True,
        transform = None, )
jpeg_path = []
jsons_path = []
extra_files_path = []
blanks_path = []
blanks_json = []
blank_extra_files_path = []

for sample in samples:
    if (min_value <= sample.get('theoreticalValue') <= max_value):
        jpeg_path.append(sample.get('fileName'))
        jsons_path.append(sample.get('fileName').replace('.jpg', '.json'))
        for extra_file in sample.get('extraFileNames'):
            extra_files_path.append(extra_file)
        if sample.get('blankFileName') not in blanks_path:
            blanks_path.append(sample.get('blankFileName'))
            blanks_json.append(sample.get('blankFileName').replace('.jpg', '.json'))
            for i in range(1,6):
                blank_extra_files_path.append(sample.get('blankFileName').replace('.jpg', f'-extra-{i}.jpg'))

print("Filtering samples")

for json_file in blanks_json:
    try:
        shutil.copy(json_file, SAVE_PATH)
    except:
        if 'extra-5' in json_file:
            pass
        else:
            print(f"File not found: {json_file} ")

for jpeg_file in blanks_path:
    try:
        shutil.copy(jpeg_file, SAVE_PATH)
    except:
        if 'extra-5' in jpeg_file:
            pass
        else:
            print(f"File not found: {jpeg_file} ")

for extra_file in blank_extra_files_path:
    try:
        shutil.copy(extra_file, SAVE_PATH)
    except:
        if 'extra-5' in extra_file:
            pass
        else:
            print(f"File not found: {extra_file} ")

for json_file in jsons_path:
    try:
        shutil.copy(json_file, SAVE_PATH)
    except:
        if 'extra-5' in json_file:
            pass
        else:
            print(f"File not found: {json_file} ")

for jpeg_file in jpeg_path:
    try:
        shutil.copy(jpeg_file, SAVE_PATH)
    except:
        if 'extra-5' in jpeg_file:
            pass
        else:
            print(f"File not found: {jpeg_file} ")

for extra_file in extra_files_path:
    try:
        shutil.copy(extra_file, SAVE_PATH)
    except:
        if 'extra-5' in extra_file:
            pass
        else:
            print(f"File not found: {extra_file} ")

# deletes cache
shutil.rmtree(CACHE_PATH)
os.rmdir(os.path.join(CACHE_BASE_DIR, f"{ANALYTE}"))
os.rmdir(CACHE_BASE_DIR)

print("Done!")