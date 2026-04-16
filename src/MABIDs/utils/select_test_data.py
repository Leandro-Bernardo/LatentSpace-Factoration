import os
import json
import random
import shutil

from typing import List
from tqdm import tqdm

# TODO carregar os dados
def load_data(path: str) -> List[str]:
    response = list()
    for root, _, files in os.walk(path):
        jsons = list(filter(lambda f: f.endswith('.json'), files))
        response += list(map(lambda f: os.path.join(root, f), jsons))
    return response

# TODO agrupar usando a referência
def group_by_reference(json_pths: List[str]) -> dict:
    refs = list()
    not_refs = list()
    for json_pth in tqdm(json_pths, desc='Spliting samp. type'):
        with open(json_pth, 'r', encoding='utf-8') as json_file:            
            data = json.load(json_file)
            ref_name = os.path.splitext(data['sample']['blankFileName'])[0] if data['sample']['blankFileName'] is not None else None
            smp_name = os.path.splitext(os.path.basename(json_pth))[0]
            item = {'pth': json_pth, 'ref': ref_name, 'smp': smp_name}
            if ref_name is None:
                refs.append(item)
            else:
                not_refs.append(item)
    groups = list()
    for ref in refs:
        response = {'ref': ref, 'samples': list()}
        matches = list(filter(lambda x: x['ref']==ref['smp'], not_refs))
        if len(matches) > 0:    
            list(map(lambda s: response['samples'].append(s), matches))
            groups.append(response)
    return groups
      
# TODO separar a quantidade de sequências para trainamento, e teste
def move_sequences(groups: List[dict], dir: str, folder_name: str) -> None:
    output_dir = os.path.join(dir, folder_name)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    for group in tqdm(groups, desc=f'Moving {folder_name}'):
        file_paths = get_all_files(group)
        list(map(lambda p: shutil.copy(p, os.path.join(output_dir, os.path.basename(p))), file_paths))

def get_all_files(sequence: dict) -> List[str]:
    jsons = [sequence['ref']['pth']]
    jsons += list(map(lambda s: s['pth'], sequence['samples']))
    found = list()
    for json in jsons:
        root, file_name = os.path.split(json)
        files = os.listdir(root)
        found += list(map(lambda x: os.path.join(root, x), list(filter(lambda f: f.__contains__(file_name[:-5]), files))))
    return found

if __name__ == '__main__':
    PATHS = [
        "Y://repo//MABIDs-Dataset-IronOxide//train", 
        "Y://repo//MABIDs-Dataset-IronOxide//test"
    ]
    N_SEQ = [50, 2]
    OUTPUT_DIR_NAME = ["train", "test"]
    
    for idx, pth in enumerate(PATHS):
        root, folder_name = os.path.split(pth)
        jsons = load_data(pth)
        groups = group_by_reference(jsons)
        
        choices = random.choices(groups, k=N_SEQ[idx])
        
        move_sequences(choices, "Y://repo//MABIDs-Dataset-IronOxide//", f"_{OUTPUT_DIR_NAME[idx]}")
    
    pass