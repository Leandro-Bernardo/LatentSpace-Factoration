import os
import wandb
import shutil
from tqdm import tqdm

ANALYTE = ""
SWEEP_ID = ""
LOCAL_PATH = rf"Y:\repo\MABIDs\python\chemical-analysis-{ANALYTE}"

def get_failed_runs(analyte, sweep_id):
    api = wandb.Api()
    sweep_path = rf"prograf-uff/chemical-analysis-{analyte}/{sweep_id}"
    sweep = api.sweep(sweep_path)
#    return sweep
    number_runs = 0
    failed_runs = set()
    print("Percorrendo runs do sweep:")
    for run in tqdm(sweep.runs):
        if run.state == "failed" or run.state == "crashed":
            failed_runs.add(run.id)
            number_runs += 1

    print(f"Foram encontradas {number_runs} runs nao concluidas")
    return failed_runs

def delete_failed_runs(failed_runs):
    if not failed_runs:
        return

    if not os.path.exists(LOCAL_PATH):
        print(f"Pasta {LOCAL_PATH} não encontrada.")
        return

    number_deletes = 0
    folders = os.listdir(LOCAL_PATH)

    print("Percorrendo pastas para deletar:")
    for folder in tqdm(folders):
        folder_path = os.path.join(LOCAL_PATH, folder)
        if folder in failed_runs:
            shutil.rmtree(folder_path)
            number_deletes += 1
    print(f"Foram deletadas {number_deletes} runs")



failed_runs = get_failed_runs(ANALYTE,SWEEP_ID)
delete_failed_runs(failed_runs)