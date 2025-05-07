import os
import numpy as np
import shutil

folder_path = '/home/shellyf/Projects/data/maze/'  # Update with your actual folder path
destination_folder = '/home/shellyf/Projects/data/maze_seed_1_19/'
level_seeds = []
count = 0
os.makedirs(destination_folder, exist_ok=True)
seed_games = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0, 8: 0, 9: 0, 10: 0, 11: 0, 12: 0, 13: 0, 14: 0, 15: 0, 16: 0, 17: 0, 18: 0, 19: 0}
for filename in os.listdir(folder_path):
    if filename.endswith('.npy'):
        parts = filename.split('_')
        if len(parts) >= 5:
            level_seed = int(parts[3])  # 0=timestamp, 1=index, 2=episode_length, 3=level_seed
            if level_seed < 20:
                level_seeds.append(level_seed)
                seed_games[level_seed] +=1
                count += 1
                # Copy the file to the destination folder
                shutil.copy(os.path.join(folder_path, filename), os.path.join(destination_folder, filename))

print(seed_games)
print("level seeds = ", level_seeds)
print("num of seeds = ", count)
print(np.where(level_seeds == '7')[0])
"""
level seeds =  186
num of seeds =  1645
"""
