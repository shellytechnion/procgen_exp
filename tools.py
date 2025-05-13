import os
import numpy as np
import shutil

def count_seeds():
    """
dict_keys([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 15, 16, 17, 18, 19, 20, 21, 22, 23, 25, 26, 27, 28, 29, 31, 32, 34, 35, 36, 37, 38, 40, 44, 45, 47, 48, 49, 50, 51, 52, 53, 54, 55, 58, 60, 61, 62, 63, 64, 65, 66, 69, 70, 71, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 86, 88, 89, 90, 91, 92, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 107, 108, 109, 110, 111, 112, 114, 115, 116, 117, 118, 119, 120, 121])
{1: 15, 2: 15, 3: 15, 4: 15, 5: 15, 6: 15, 7: 15, 8: 15, 9: 15, 10: 15, 11: 15, 12: 15, 13: 15, 15: 15, 16: 15, 17: 15, 18: 15, 19: 15, 20: 15, 21: 15, 22: 15, 23: 15, 25: 15, 26: 15, 27: 15, 28: 15, 29: 15, 31: 15, 32: 15, 34: 15, 35: 15, 36: 15, 37: 15, 38: 15, 40: 15, 44: 15, 45: 15, 47: 15, 48: 15, 49: 15, 50: 15, 51: 15, 52: 15, 53: 15, 54: 15, 55: 15, 58: 15, 60: 15, 61: 15, 62: 15, 63: 15, 64: 15, 65: 15, 66: 15, 69: 15, 70: 15, 71: 15, 73: 15, 74: 15, 75: 15, 76: 15, 77: 15, 78: 15, 79: 15, 80: 15, 81: 15, 82: 15, 83: 15, 84: 15, 86: 15, 88: 15, 89: 15, 90: 15, 91: 15, 92: 15, 94: 15, 95: 15, 96: 15, 97: 15, 98: 15, 99: 15, 100: 15, 101: 15, 102: 15, 103: 15, 104: 15, 107: 15, 108: 15, 109: 15, 110: 15, 111: 15, 112: 15, 114: 15, 115: 15, 116: 15, 117: 15, 118: 15, 119: 15, 120: 15, 121: 15}
num of seeds =  1500
    """
    folder_path = '/home/shellyf/Projects/data/data_ppo/maze_full/'  # Update with your actual folder path
    # folder_path = '/home/shellyf/Projects/data/data_blindfolded_expert/maze'  # Update with your actual folder path
    # folder_path = '/home/shellyf/Projects/data/data_ppo/maze_full'  # Update with your actual folder path
    destination_folder = '/home/shellyf/Projects/data/data_ppo/maze_100_seed_20_traj'
    level_seeds = []
    count = 0
    os.makedirs(destination_folder, exist_ok=True)
    seed_games = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0, 8: 0, 9: 0, 10: 0,
                  11: 0, 12: 0, 13: 0, 15: 0, 16: 0, 17: 0, 18: 0, 19: 0,
                  20: 0, 21: 0, 22: 0, 23: 0, 25: 0, 26: 0, 27: 0, 28: 0,
                  29: 0, 31: 0, 32: 0, 34: 0, 35: 0, 36: 0, 37: 0,
                  38: 0, 40: 0, 44: 0, 45: 0,
                  47: 0, 48: 0, 49: 0, 50: 0, 51: 0, 52: 0, 53: 0, 54: 0, 55: 0,
                  58: 0, 60: 0, 61: 0, 62: 0, 63: 0, 64: 0,
                  65: 0, 66: 0, 69: 0, 70: 0, 71: 0, 73: 0,
                  74: 0, 75: 0, 76: 0, 77: 0, 78: 0, 79: 0, 80: 0, 81: 0, 82: 0,
                  83: 0, 84: 0, 86: 0, 88: 0, 89: 0, 90: 0, 91: 0,
                  92: 0, 94: 0, 95: 0, 96: 0, 97: 0, 98: 0, 99: 0,
                  100: 0, 101: 0, 102: 0, 103: 0, 104: 0, 107: 0,
                  108: 0, 109: 0, 110: 0, 111: 0, 112: 0, 114: 0, 115: 0,
                  116: 0, 117: 0, 118: 0, 119: 0, 120: 0, 121: 0}

    # seed_games = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 7: 0, 8: 0, 9: 0, 10: 0,
    #               11: 0, 12: 0, 13: 0, 15: 0, 16: 0, 17: 0, 18: 0, 19: 0,
    #               20: 0, 21: 0, 22: 0, 23: 0, 25: 0, 26: 0, 27: 0,
    #               29: 0, 31: 0, 32: 0, 35: 0, 36: 0,
    #               40: 0, 44: 0, 45: 0,
    #               47: 0, 48: 0, 51: 0, 52: 0, 53: 0, 54: 0,
    #               58: 0, 60: 0, 61: 0, 62: 0, 63: 0, 64: 0,
    #               65: 0, 66: 0, 69: 0, 70: 0, 71: 0,
    #               74: 0, 75: 0, 76: 0, 77: 0, 78: 0, 80: 0, 81: 0, 82: 0,
    #               83: 0, 84: 0, 86: 0, 88: 0, 89: 0, 90: 0, 91: 0,
    #               92: 0, 94: 0, 95: 0, 96: 0, 97: 0, 98: 0, 99: 0,
    #               100: 0, 101: 0, 102: 0, 103: 0, 107: 0,
    #               108: 0, 110: 0, 111: 0, 112: 0, 114: 0, 115: 0,
    #               116: 0, 117: 0, 118: 0, 119: 0, 120: 0, 121: 0}
    # [6, 28, 34, 37, 38, 49, 50, 55, 73, 79, 104, 109]

    print(seed_games.keys())
    total_eposode_length = 0
    for filename in os.listdir(folder_path):
        if filename.endswith('.npy'):
            parts = filename.split('_')
            if len(parts) >= 5:
                level_seed = int(parts[3])  # 0=timestamp, 1=index, 2=episode_length, 3=level_seed
                episode_length = int(parts[2])
                if level_seed in seed_games.keys() and episode_length < 500 and seed_games[level_seed] < 20: # and seed_games[level_seed] < 15
                    level_seeds.append(level_seed)
                    seed_games[level_seed] +=1
                    total_eposode_length += episode_length
                    # Copy the file to the destination folder
                    # shutil.copy(os.path.join(folder_path, filename), os.path.join(destination_folder, filename))

    # count seeds with 15 or less
    needed_seeds = []
    for seed in seed_games:
        if seed_games[seed] < 20:
            # print(seed)
            needed_seeds.append(seed)

    print(seed_games)
    # print("level seeds = ", level_seeds)
    print("num of seeds = ", count)
    print(sum(seed_games.values()))
    print("total episode length = ", total_eposode_length)
    print("needed_seeds = ", needed_seeds)


def add_index():
    folder_path = '/home/shellyf/Projects/data/data_blindfolded_expert/maze'
    # folder_path = '/home/shellyf/Projects/data/data_blindfolded_expert/data_shir/procgen_data'

    # List all files in the directory
    files = sorted([f for f in os.listdir(folder_path) if f.endswith('.npy') and "202505" in f.split("_")[0]])

    # Iterate through the files and rename them
    for index, filename in enumerate(files, start=1840):
        time = filename.split("_")[0]
        if "202505" not in time:
            continue
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

def transfer_to_with_backgrounds():
    """
    Example random agent script using the gym3 API to demonstrate that procgen works
    """
    folder_path = '/home/shellyf/Projects/data/procgen_disable-backgrounds'
    from gym3 import types_np
    from procgen import ProcgenGym3Env
    import copy

    files = [f for f in os.listdir(folder_path) if f.endswith('.npy')]
    #f"{timestamp}_{self.count_actions}_{self.seed}_{int(episode_ret)}.npy")
    kwargs = {
        "num_levels": 1,
        "use_backgrounds": True,
        "distribution_mode": "easy"
    }
    for file in files:
        _, _, count_actions, level, episode_ret = file.split("_")
        kwargs["start_level"] = int(level)
        # Load the .npy file
        file_path = os.path.join(folder_path, file)
        data_dict = np.load(file_path, allow_pickle=True).item()

        # Extract data
        old_observations = copy.deepcopy(data_dict["observations"])
        data_dict["observations"] = []
        actions = data_dict["actions"]
        rewards = data_dict["rewards"]
        dones = data_dict["dones"]

        env = ProcgenGym3Env(num=1, env_name="maze", **kwargs)
        rew, obs, first = env.observe()
        img = copy.deepcopy(obs["rgb"])
        img = img.squeeze()
        img = np.transpose(img, (2, 0, 1))
        data_dict["observations"].append(img)

        # Roll the data into the environment
        for act, reward, done in zip(actions, rewards, dones):
            env.act(act)
            rew, obs, first = env.observe()
            img = copy.deepcopy(obs["rgb"])
            img = img.squeeze()
            img = np.transpose(img, (2, 0, 1))
            data_dict["observations"].append(img)

            print(f"Reward: {rew}, Done: {done}")
            # if first:
            #     break

        new_filepath = os.path.join(folder_path.replace("procgen_disable-backgrounds", "procgen"), file)
        np.save(new_filepath, data_dict)
        print("saved data_dict to {}".format(new_filepath))

if __name__ == "__main__":
    count_seeds()
    # remove_index()
    # add_index()
    # transfer_to_with_backgrounds()
    # data_dict = np.load(r"/home/shellyf/Projects/data/843_20250509225510_160_64_10.npy", allow_pickle=True).item()
    # print(data_dict)




