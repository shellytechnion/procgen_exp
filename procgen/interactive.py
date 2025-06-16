#!/usr/bin/env python
import argparse
import cv2
import numpy as np
from typing import Optional
from procgen import ProcgenGym3Env
import copy
from datetime import datetime
import os
import random
from tqdm import tqdm

try:
    from .env import ENV_NAMES
except ImportError:
    from procgen.env import ENV_NAMES
from gym3 import Interactive, VideoRecorderWrapper, unwrap


RECORD_DIR = None
ENV = "maze"  # default environment, can be overridden by command line argument
NAME = "None"

class ProcgenInteractive(Interactive):
    def __init__(self, seed, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._saved_state = None
        self.env = args[0]
        self.seed = seed
        self.count_actions = 0
        self.data_dict = {"observations": [], "actions": [], "rewards": [], "dones": []}

    def _update(self, dt, keys_clicked, keys_pressed):
        if "LEFT_SHIFT" in keys_pressed and "F1" in keys_clicked:
            print("save state")
            self._saved_state = unwrap(self._env).get_state()
        elif "F1" in keys_clicked:
            print("load state")
            if self._saved_state is not None:
                unwrap(self._env).set_state(self._saved_state)
        # Save the current observation and action
        keys = keys_clicked if self._synchronous else keys_pressed
        act = self._keys_to_act(keys)
        if self.count_actions == 0:
            t1, obs1, batch_first = self.env.observe()
            img = copy.deepcopy(obs1["rgb"])
            img = img.squeeze()
            img = np.transpose(img, (2, 0, 1))
            # cv2.imwrite(str(RECORD_DIR) + "/{}_{}_{}_{}.png".format(self.seed, self.count_actions, self._episode_return, (self._episode_return == 10.0)), img)
            self.data_dict["observations"].append(img)
            self.count_actions+=1
        if act[0] == 4: # 4 is do nothing
            return
        # Call the original update method
        #print("act", act)
        super()._update(dt, keys_clicked, keys_pressed)
        t1, obs1, batch_first = self.env.observe()
        if self._last_info["episode_return"] == 10.0 or self.count_actions > 499 or batch_first:
            print("episode return is 10.0")
            img = obs1["rgb"]
            img = img.squeeze()
            # cv2.imwrite(
            #     str(RECORD_DIR) +"/{}_{}_{}_{}.png".format(self.seed, self.count_actions, self._episode_return, True), img)
            self.data_dict["observations"].append(np.transpose(img, (2, 0, 1)))
            self.data_dict["actions"].append(act)
            self.data_dict["rewards"].append(self._last_info["episode_return"])
            self.data_dict["dones"].append(1)
            self._renderer._should_close = True
            self._renderer.finish()
            # Save data_dict to a .npy file
            timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
            episode_ret = self._last_info["episode_return"]
            filename = os.path.join(RECORD_DIR, f"{NAME}_{timestamp}_{self.count_actions}_{self.seed}_{int(episode_ret)}.npy")
            np.save(filename, self.data_dict)
            print("saved data_dict to {}".format(filename))

        elif act[0] != 4: # 4 is do nothing
            img = obs1["rgb"]
            img = img.squeeze()
            # cv2.imwrite(str(RECORD_DIR) + "/{}_{}_{}_{}.png".format(self.seed, self.count_actions, self._episode_return, (self._episode_return == 10.0)), img)
            self.data_dict["observations"].append(np.transpose(img, (2, 0, 1)))
            self.data_dict["actions"].append(act)
            self.data_dict["rewards"].append(self._episode_return)
            self.data_dict["dones"].append(1 if self._episode_return == 10 else 0)
            self.count_actions+=1

    def _get_image(self) -> Optional[np.ndarray]:
        """
        Get the image that we should display to the user for the current step,
        with a mask applied.
        """
        global ENV
        _, ob, _ = self._env.observe()
        if self._info_key is None:
            if self._ob_key is not None:
                ob = ob[self._ob_key]
            image = ob[0]
        else:
            info = self._env.get_info()
            image = info[0].get(self._info_key)

        if image is not None:
            # Extract agent position from the info dictionary
            # info = self._env.get_info()[0]
            h, w, _ = image.shape
            # Find the first pixel with the color (187, 203, 204)
            if ENV == "maze":
                target_color = np.array([187, 203, 204])
                radius = min(h, w) // 8  # Adjust radius as needed
            elif ENV  == "heist":
                target_color = np.array([255, 255, 255])
                radius = min(h, w) // 6  # Adjust radius as needed
            elif ENV == "jumper":
                target_color = np.array([188, 113, 224])
                radius = min(h, w) // 8  # Adjust radius as needed
            else:
                # throw an error or handle other environments
                raise ValueError(f"Unsupported environment: {ENV}")
            match = np.all(image == target_color, axis=-1)
            coords = np.argwhere(match)
            agent_pos = coords[0]

            if agent_pos is not None:
                # Map agent position to pixel coordinates

                center_x = int(agent_pos[1])
                center_y = int(agent_pos[0])

                # Create a circular mask around the agent
                mask = np.zeros((h, w), dtype=np.uint8)
                # radius = min(h, w) // 8  # Adjust radius as needed
                cv2.circle(mask, (center_x, center_y), radius, 255, -1)

                # Apply the mask to the image
                masked_image = cv2.bitwise_and(image, image, mask=mask)
                return masked_image

        return image


def make_interactive(seed, vision, record_dir, **kwargs):
    global RECORD_DIR, NAME
    info_key = None
    ob_key = None
    if vision == "human":
        info_key = "rgb"
        kwargs["render_mode"] = "rgb_array"
    else:
        ob_key = "rgb"

    env = ProcgenGym3Env(num=1, **kwargs)
    if record_dir is not None:
        RECORD_DIR = record_dir
    #     env = VideoRecorderWrapper(
    #         env=env, directory=record_dir, ob_key=ob_key, info_key=info_key
    #     )
    h, w, _ = env.ob_space["rgb"].shape
    return ProcgenInteractive(
        seed, env,
        ob_key=ob_key,
        info_key=info_key,
        width=w * 12,
        height=h * 12,
        synchronous=True
    )


def main():
    global ENV, NAME
    default_str = "(default: %(default)s)"
    parser = argparse.ArgumentParser(
        description="Interactive version of Procgen allowing you to play the games"
    )
    parser.add_argument(
        "--vision",
        default="human",
        choices=["agent", "human"],
        help="level of fidelity of observation " + default_str,
    )
    parser.add_argument("--record-dir", help="directory to record moinfo_keys to")
    parser.add_argument(
        "--distribution-mode",
        default="hard",
        help="which distribution mode to use for the level generation " + default_str,
    )
    parser.add_argument(
        "--env-name",
        default="coinrun",
        help="name of game to create " + default_str,
        choices=ENV_NAMES + ["coinrun_old"],
    )
    parser.add_argument(
        "--level-seed", type=int, help="select an individual level to use"
    )
    parser.add_argument(
        "--name",
        default="None",
        help="Your name",
    )
    advanced_group = parser.add_argument_group("advanced optional switch arguments")
    advanced_group.add_argument(
        "--paint-vel-info",
        action="store_true",
        default=False,
        help="paint player velocity info in the top left corner",
    )
    advanced_group.add_argument(
        "--use-generated-assets",
        action="store_true",
        default=False,
        help="use randomly generated assets in place of human designed assets",
    )
    advanced_group.add_argument(
        "--uncenter-agent",
        action="store_true",
        default=False,
        help="display the full level for games that center the observation to the agent",
    )
    advanced_group.add_argument(
        "--disable-backgrounds",
        action="store_true",
        default=False,
        help="disable human designed backgrounds",
    )
    advanced_group.add_argument(
        "--restrict-themes",
        action="store_true",
        default=False,
        help="restricts games that use multiple themes to use a single theme",
    )
    advanced_group.add_argument(
        "--use-monochrome-assets",
        action="store_true",
        default=False,
        help="use monochromatic rectangles instead of human designed assets",
    )

    args = parser.parse_args()

    kwargs = {
        "paint_vel_info": args.paint_vel_info,
        "use_generated_assets": args.use_generated_assets,
        "center_agent": not args.uncenter_agent,
        "use_backgrounds": not args.disable_backgrounds,
        "restrict_themes": args.restrict_themes,
        "use_monochrome_assets": args.use_monochrome_assets,
    }
    if args.env_name != "coinrun_old":
        kwargs["distribution_mode"] = args.distribution_mode
    if args.level_seed is not None:
        kwargs["start_level"] = args.level_seed
        kwargs["num_levels"] = 1
    if args.name is not "None":
        NAME = args.name

    # create a list of all the levels (seeds)
    # levels = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 15, 16, 17, 18, 19, 20, 21, 22, 23, 25, 26, 27, 28, 29, 31, 32, 34, 35, 36, 37, 38, 40, 44, 45, 47, 48, 49, 50, 51, 52, 53, 54, 55, 58, 60, 61, 62, 63, 64, 65, 66, 69, 70, 71, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 86, 88, 89, 90, 91, 92, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 107, 108, 109, 110, 111, 112, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123]
    # levels = [76]
    if args.env_name == "maze":
        ENV = "maze"
        # len = 166
        levels = [1, 2]
        # levels =  [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 15, 16, 17, 18, 19, 20, 21, 22, 23, 25, 26,
        #                27, 28, 29, 31, 32, 34, 35, 36, 37, 38, 40, 44, 45, 47, 48, 49, 50, 51, 52, 53, 54, 55,
        #                58, 60, 61, 62, 63, 64, 65, 66, 69, 70, 71, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83,
        #                84, 86, 88, 89, 90, 91, 92, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 107, 108, 109, 110,
        #                111, 112, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 125, 128, 129, 130, 131, 132, 134,
        #                135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150, 151, 152, 153,
        #                155, 156, 157, 158, 159, 160, 161, 162, 163, 164, 165, 166, 167, 168, 170, 171, 172, 173, 174,
        #                175, 176, 177, 180, 181, 182, 183, 184, 186, 188, 189, 190, 191, 192, 193, 195, 196, 197, 198, 199]
    elif args.env_name == "heist":
        ENV = "heist"
        # len = 114
        levels = [1, 2]
        # levels =  [0, 2, 5, 6, 8, 10, 11, 12, 13, 19, 20, 23, 24, 25, 28, 30, 31, 33, 35, 36, 38, 39, 40, 42, 43, 45, 46,
        #            47, 48, 49, 50, 53, 54, 55, 58, 62, 63, 64, 66, 67, 68, 70, 71, 72, 73, 74, 75, 76, 77, 78, 80, 84, 86,
        #            87, 88, 89, 90, 91, 93, 96, 103, 105, 106, 107, 108, 109, 110, 111, 113, 116, 117, 118, 119, 121, 122,
        #            123, 127, 128, 129, 135, 137, 138, 139, 140, 142, 144, 146, 150, 151, 152, 153, 157, 158, 160, 165, 168,
        #            169, 171, 173, 175, 176, 177, 178, 179, 180, 181, 182, 183, 185, 186, 187, 188, 195, 197]
    elif args.env_name == "jumper":
        ENV = "jumper"
        # len = 177
        levels = [1,2]
        # levels = [1, 2, 4, 6, 7, 8, 10, 12, 13, 14, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 28, 29, 30, 31, 32, 33,
        #              34, 35, 36, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59,
        #              60, 61, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85,
        #              87, 89, 90, 91, 92, 93, 94, 95, 97, 98, 100, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112,
        #              114, 115, 117, 118, 119, 120, 121, 122, 123, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135,
        #              136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 155, 156,
        #              157, 158, 159, 160, 161, 162, 163, 165, 166, 167, 168, 170, 171, 172, 173, 174, 175, 177, 178, 179,
        #              180, 181, 183, 184, 185, 186, 187, 188, 189, 190, 191, 192, 193, 194, 195, 196, 197, 198, 199]
    random.shuffle(levels)
    for level in tqdm(levels):
        kwargs["start_level"] = level
        kwargs["num_levels"] = 1
        # create a directory for each level
        # record_dir = os.path.join(args.record_dir, str(level))
        # if not os.path.exists(args.record_dir):
        #     os.makedirs(args.record_dir)
        # create the interactive environment
        print("record_dir", args.record_dir)
        ia = make_interactive(level,
            args.vision, record_dir=args.record_dir, env_name=args.env_name, **kwargs
        )
        try:
            ia.run()
        except Exception as e:
            continue


if __name__ == "__main__":
    main()
