import os
import numpy as np
import shutil

def count_seeds():
    """
    level seeds =  186
    num of seeds =  1645
    """
    folder_path = '/home/shellyf/Projects/data/maze_full/'  # Update with your actual folder path
    destination_folder = '/home/shellyf/Projects/data/maze/'
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
                    # shutil.copy(os.path.join(folder_path, filename), os.path.join(destination_folder, filename))

    print(seed_games)
    print("level seeds = ", level_seeds)
    print("num of seeds = ", count)
    print(np.where(level_seeds == '7')[0])

def add_index():
    folder_path = '/home/shellyf/Projects/data/procgen'

    # List all files in the directory
    files = sorted([f for f in os.listdir(folder_path) if f.endswith('.npy')])

    # Iterate through the files and rename them
    for index, filename in enumerate(files, start=344):
        old_path = os.path.join(folder_path, filename)
        if os.path.isfile(old_path) and filename.endswith("npy"):  # Ensure it's a file
            new_filename = f"{index}_{filename}"
            new_path = os.path.join(folder_path, new_filename)
            os.rename(old_path, new_path)

    print("Renaming completed.")

def remove_index():
    folder_path = '/home/shellyf/Projects/data/procgen'

    # List all files in the directory
    files = [f for f in os.listdir(folder_path) if f.endswith('.npy')]

    # Iterate through the files and rename them
    for filename in files:
        old_path = os.path.join(folder_path, filename)
        if os.path.isfile(old_path):
            # Remove the index by splitting at the first underscore
            new_filename = '_'.join(filename.split('_')[1:])
            new_path = os.path.join(folder_path, new_filename)
            os.rename(old_path, new_path)

    print("Index removal completed.")

if __name__ == "__main__":
    # count_seeds()
    # remove_index()
    add_index()



