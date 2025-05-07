import os
import numpy as np
import shutil

folder_path = '/home/shellyf/Projects/data/maze/'  # Update with your actual folder path
destination_folder = '/home/shellyf/Projects/data/maze_seed_1_19/'
level_seeds = []
count = 0
os.makedirs(destination_folder, exist_ok=True)

for filename in os.listdir(folder_path):
    if filename.endswith('.npy'):
        parts = filename.split('_')
        if len(parts) >= 5:
            level_seed = int(parts[3])  # 0=timestamp, 1=index, 2=episode_length, 3=level_seed
            if level_seed < 20:
                level_seeds.append(level_seed)
                count += 1
                # Copy the file to the destination folder
                shutil.copy(os.path.join(folder_path, filename), os.path.join(destination_folder, filename))

print("level seeds = ", level_seeds)
print("num of seeds = ", count)
print(np.where(level_seeds == '7')[0])
"""
level seeds =  186
num of seeds =  1645
"""
