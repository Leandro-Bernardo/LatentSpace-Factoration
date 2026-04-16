import os
import numpy as np
import chemical_analysis as ca
import matplotlib.pyplot as plt
import cv2
from tqdm import tqdm

### mover esse script para a pasta anterior (../MABIDS/python/) para poder importar o módulo chemical_analysis ###

ANALYTE = "iron_oxid"
PATH_DIR = [f"G:\\Dataset\\{ANALYTE.capitalize()}\\IronOxidTrainingSamples"]
CACHE_PATH = f"G:\\Dataset\\{{ANALYTE.capitalize()}}\\cache"
SAVE_PATH = f"G:\\Dataset\\{ANALYTE.capitalize()}\\roi_analysis_results"

os.makedirs(CACHE_PATH, exist_ok=True)
os.makedirs(SAVE_PATH, exist_ok=True)
processing = {"alkalinity": {"samples": ca.alkalinity.AlkalinitySampleDataset,
                             "processed_samples": ca.alkalinity.ProcessedAlkalinitySampleDataset},
              "chloride"  : {"samples": ca.chloride.ChlorideSampleDataset,
                             "processed_samples": ca.chloride.ProcessedChlorideSampleDataset},
              "phosphate" : {"samples": ca.phosphate.PhosphateSampleDataset,
                             "processed_samples": ca.phosphate.ProcessedPhosphateSampleDataset},
              "bisulfite" : {"samples": ca.bisulfite2d.Bisulfite2DSampleDataset,
                             "processed_samples": ca.bisulfite2d.ProcessedBisulfite2DSampleDataset},
            #   "emulsion": {"samples": ca.emulsion.EmulsionSampleDataset,
            #                  "processed_samples": ca.emulsion.ProcessedEmulsionSampleDataset},
              "iron_oxid" : {"samples": ca.iron_oxid.IronOxidSampleDataset,
                             "processed_samples": ca.iron_oxid.ProcessedIronOxidSampleDataset},
              "iron2"     : {"samples": ca.iron2.Iron2SampleDataset,
                             "processed_samples": ca.iron2.ProcessedIron2SampleDataset},
            #   "iron3": {"samples": ca.iron3.Iron3SampleDataset,
            #                  "processed_samples": ca.iron3.ProcessedIron3SampleDataset},
              "ph"        : {"samples": ca.ph.PhSampleDataset,
                             "processed_samples": ca.ph.ProcessedPhSampleDataset},
              "redox"     : {"samples": ca.redox.RedoxSampleDataset,
                             "processed_samples": ca.redox.ProcessedRedoxSampleDataset},
            #   "suspended" : {"samples": ca.suspended.SuspendedSampleDataset,
            #                  "processed_samples": ca.suspended.ProcessedSuspendedSampleDataset},

}

pca_stats = {   "alkalinity": {"lab_mean": None, "lab_sorted_eigenvectors": None},
                #"chloride"  : {"lab_mean": np.load(ca.chloride.PCA_STATS)['lab_mean']  , "lab_sorted_eigenvectors": np.load(ca.chloride.PCA_STATS)['lab_sorted_eigenvectors']},
                #"sulfate"   : {"lab_mean": np.load(ca.sulfate.PCA_STATS)['lab_mean']   , "lab_sorted_eigenvectors": np.load(ca.sulfate.PCA_STATS)['lab_sorted_eigenvectors']},
                #"phosphate" : {"lab_mean": np.load(ca.phosphate.PCA_STATS)['lab_mean'] , "lab_sorted_eigenvectors": np.load(ca.phosphate.PCA_STATS)['lab_sorted_eigenvectors']},
                #"bissulfite2d": {"lab_mean": np.load(ca.bisulfite2d.PCA_STATS)['lab_mean'] , "lab_sorted_eigenvectors": np.load(ca.bisulfite2d.PCA_STATS)['lab_sorted_eigenvectors']},
                "iron_oxid" : {"lab_mean": np.load(ca.iron_oxid.PCA_STATS)['lab_mean'] , "lab_sorted_eigenvectors": np.load(ca.iron_oxid.PCA_STATS)['lab_sorted_eigenvectors']},
                }

sample_dataset = processing[ANALYTE]["samples"]
processed_sample_dataset = processing[ANALYTE]["processed_samples"]

# data preprocessing
samples = sample_dataset(
        base_dirs = PATH_DIR,
        progress_bar = True,
        skip_blank_samples = False,
        skip_incomplete_samples = True,
        skip_inference_sample= True,
        skip_training_sample = False,
        verbose = True
    )
processed_samples = processed_sample_dataset(
            dataset = samples,
            cache_dir = CACHE_PATH,
            num_augmented_samples = 0,
            progress_bar = True,
            transform = None,
            lab_mean= pca_stats[f"{ANALYTE}"]['lab_mean'],
            lab_sorted_eigenvectors = pca_stats[f"{ANALYTE}"]['lab_sorted_eigenvectors'])

for i in tqdm(range(len(processed_samples))):
  fig,ax = plt.subplots(nrows=1,ncols=3)
  fig.suptitle(f"sample:{processed_samples[i].sample_prefix}")
  ax[0].imshow(cv2.cvtColor(processed_samples[i].sample_bgr_image, cv2.COLOR_BGR2RGB)/255)
  ax[1].imshow(processed_samples[i].sample_analyte_mask)
  ax[2].imshow(cv2.cvtColor(processed_samples[i].sample_bgr_image*np.expand_dims(processed_samples[i].sample_analyte_mask,axis=2), cv2.COLOR_BGR2RGB)/255)
  plt.savefig(f"{SAVE_PATH}\\{processed_samples[i].sample_prefix}.png")
  plt.close('all')

