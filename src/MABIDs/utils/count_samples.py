from tqdm import tqdm
from typing import Any, Dict, List, Tuple
import json, os, shutil


path_dir = [fr"F:\this\A25\adapted-jsons-type1"]

def count_jsons(base_dirs: List[str]) -> Tuple[Dict[str, Any], Dict[str, str], List[str]]:
    json_count = 0
    img_count = 0
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
                        json_count+= 1
                    elif filename.lower().endswith(".jpg"):
                        img_count +=1
                elif os.path.isdir(path):
                    not_visited_dirs.append(path)
                    pbar.total += 1
                    pbar.refresh()
            pbar.update(1)
    return json_count, img_count

if __name__ == "__main__":
    jsons, images = count_jsons(path_dir)

print(f"jsons: {jsons}, images: {images}")