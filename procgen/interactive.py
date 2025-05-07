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

try:
    from .env import ENV_NAMES
except ImportError:
    from procgen.env import ENV_NAMES
from gym3 import Interactive, VideoRecorderWrapper, unwrap


RECORD_DIR = None

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
        super()._update(dt, keys_clicked, keys_pressed)
        if self._last_info["episode_return"] == 10.0 or self.count_actions > 499:
            print("episode return is 10.0")
            t1, obs1, batch_first = self.env.observe()
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
            filename = os.path.join(RECORD_DIR, f"{timestamp}_{self.count_actions}_{self.seed}_{int(episode_ret)}.npy")
            np.save(filename, self.data_dict)
            print("saved data_dict to {}".format(filename))

        elif act[0] != 4: # 4 is do nothing
            t1, obs1, batch_first = self.env.observe()
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
            info = self._env.get_info()[0]
            # Find the first pixel with the color (187, 203, 204)
            target_color = np.array([187, 203, 204])
            match = np.all(image == target_color, axis=-1)
            coords = np.argwhere(match)
            agent_pos = coords[0]

            if agent_pos is not None:
                # Map agent position to pixel coordinates
                h, w, _ = image.shape
                center_x = int(agent_pos[1])
                center_y = int(agent_pos[0])

                # Create a circular mask around the agent
                mask = np.zeros((h, w), dtype=np.uint8)
                radius = min(h, w) // 8  # Adjust radius as needed
                cv2.circle(mask, (center_x, center_y), radius, 255, -1)

                # Apply the mask to the image
                masked_image = cv2.bitwise_and(image, image, mask=mask)
                return masked_image

        return image


def make_interactive(seed, vision, record_dir, **kwargs):
    global RECORD_DIR
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

    # create a list of all the levels ( seeds) 1-19
    levels = list(range(16, 19))
    random.shuffle(levels)
    for level in levels:
        kwargs["start_level"] = level
        kwargs["num_levels"] = 1
        # create a directory for each level
        # record_dir = os.path.join(args.record_dir, str(level))
        if not os.path.exists(args.record_dir):
            os.makedirs(args.record_dir)
        # create the interactive environment
        ia = make_interactive(args.level_seed,
            args.vision, record_dir=args.record_dir, env_name=args.env_name, **kwargs
        )
        try:
            ia.run()
        except Exception as e:
            continue


if __name__ == "__main__":
    main()
