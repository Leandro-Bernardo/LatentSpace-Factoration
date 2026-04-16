import chemical_analysis as ca
import os
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from itertools import cycle
color_cycle = cycle(plt.cm.Paired.colors)

ANALYTE = ""
PATH_DIR = [fr""]
CACHE_PATH = fr""
SAVE_PATH = os.path.join(os.path.dirname(__file__), "..",  "..", f"lab_analysis_results, {ANALYTE}")

os.makedirs(CACHE_PATH, exist_ok=True)
os.makedirs(SAVE_PATH, exist_ok=True)

dataset = {
            "alkalinity":{"dataset": ca.alkalinity.AlkalinitySampleDataset, "processed_dataset": ca.alkalinity.ProcessedAlkalinitySampleDataset},
            "chloride": {"dataset": ca.chloride.ChlorideSampleDataset, "processed_dataset": ca.chloride.ProcessedChlorideSampleDataset},
            "sulfate": {"dataset": ca.sulfate.SulfateSampleDataset, "processed_dataset": ca.sulfate.ProcessedSulfateSampleDataset},
            "sulfate2d": {"dataset": ca.sulfate2d.Sulfate2DSampleDataset, "processed_dataset": ca.sulfate2d.ProcessedSulfate2DSampleDataset},
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
            "sulfate2d"  : {"lab_mean": np.load(ca.sulfate2d.PCA_STATS)['lab_mean']  , "lab_sorted_eigenvectors": np.load(ca.sulfate2d.PCA_STATS)['lab_sorted_eigenvectors']},
            "iron2"  : {"lab_mean": np.load(ca.iron2.PCA_STATS)['lab_mean']  , "lab_sorted_eigenvectors": np.load(ca.iron2.PCA_STATS)['lab_sorted_eigenvectors']},
            "iron3"  : {"lab_mean": np.load(ca.iron3.PCA_STATS)['lab_mean']  , "lab_sorted_eigenvectors": np.load(ca.iron3.PCA_STATS)['lab_sorted_eigenvectors']},
            #"iron_oxid"  : {"lab_mean": np.load(ca.iron_oxid.PCA_STATS)['lab_mean']  , "lab_sorted_eigenvectors": np.load(ca.iron_oxid.PCA_STATS)['lab_sorted_eigenvectors']},
            "phosphate"  : {"lab_mean": np.load(ca.phosphate.PCA_STATS)['lab_mean']  , "lab_sorted_eigenvectors": np.load(ca.phosphate.PCA_STATS)['lab_sorted_eigenvectors']},
            }
SampleDataset = dataset[f"{ANALYTE.lower()}"]["dataset"]
ProcessedSampleDataset = dataset[f"{ANALYTE.lower()}"]["processed_dataset"]

#data preprocessing
samples = SampleDataset(
    base_dirs = PATH_DIR,
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

assert len(samples) == len(processed_samples)

# maps all samples devices
Samples_devices = {}
for i in tqdm(range(len(samples)), desc='mapping samples devices'):
    model = samples[i].get('device', {}).get('model')
    stock = samples[i].get('sourceStock', {}).get('name')

    if model not in Samples_devices:
        Samples_devices[model] = {}

    if stock not in Samples_devices[model]:
        Samples_devices[model][stock] = [i]
    else:
        Samples_devices[model][stock].append(i)

# groupby (by devices) all processed samples
ProcessedSamples_devices = {}
for device in Samples_devices.keys():
    ProcessedSamples_devices[device] = {key: [] for key in Samples_devices.get(device).keys()}

for device in tqdm(ProcessedSamples_devices.keys(), desc='mapping processed samples devices'):
    for solution in Samples_devices.get(device).keys():
        for i in Samples_devices.get(device).get(solution):
            ProcessedSamples_devices.get(device).get(solution).append(processed_samples[i])

A, B = {}, {}
for device in ProcessedSamples_devices.keys():
        A[device] = {}
        B[device] = {}
        for solution in ProcessedSamples_devices.get(device).keys():
                A[device][solution] = []
                B[device][solution] = []

for device in tqdm(ProcessedSamples_devices.keys(), desc='processing samples AB by device'):
    for solution in ProcessedSamples_devices.get(device).keys():
        for sample in ProcessedSamples_devices.get(device).get(solution):
            lab = sample.sample_lab_image[sample.sample_analyte_mask]

            A.get(device).get(solution).extend(lab[:, 1])
            B.get(device).get(solution).extend(lab[:, 2])


# adds a small noise to make data mose visible
displacement_direction = 1   # moves data (left or right)
displacement_X_distance = 0.0 # distance of movement in X axis (initial: 0.0)
displacement_Y_distance = 0.0 # distance of movement in Y axis (initial: 0.0)
for device in ProcessedSamples_devices.keys():
    for solution in ProcessedSamples_devices.get(device).keys():
        A[device][solution] = np.array(A.get(device).get(solution))-(displacement_direction*displacement_X_distance)
        B[device][solution] = np.array(B.get(device).get(solution))-(displacement_direction*displacement_Y_distance)
        displacement_direction *= -1
        displacement_X_distance += 0.07
        displacement_Y_distance += 0.07
marker_size = 5

fig, ax = plt.subplots(figsize=(20,20))

for device in ProcessedSamples_devices.keys():
    for solution in ProcessedSamples_devices.get(device).keys():
        ax.scatter(A.get(device).get(solution), B.get(device).get(solution), marker='.', s=marker_size, label=f'{device}', color = next(color_cycle))#, c=color_map[device])

        marker_size+=3
ax.set_title(f'Grafico AxB discretizado por celular e solução', fontsize=16)

ax.set_xlabel('A')
ax.set_ylabel('B')
ax.set_xlim([-128, 128])
ax.set_ylim([-128, 128])
ax.legend(fontsize=20, scatterpoints=3, markerscale=2)
plt.savefig(f"{SAVE_PATH}\\plot.png", dpi=300)
