import datetime
import glob
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import tyro

from gello.agents.agent import BimanualAgent, DummyAgent
from gello.agents.gello_agent import GelloAgent
from gello.data_utils.format_obs import save_frame
from gello.env import RobotEnv
from gello.robots.robot import PrintRobot
from gello.zmq_core.robot_node import ZMQClientRobot


import argparse
import concurrent.futures
import json
import logging
import os
import platform
import shutil
import time
import traceback
from contextlib import nullcontext
from functools import cache
from pathlib import Path

import cv2
import torch
import tqdm
from omegaconf import DictConfig
from PIL import Image
from termcolor import colored
from lerobot.common.datasets.compute_stats import compute_stats
from lerobot.common.datasets.lerobot_dataset import CODEBASE_VERSION, LeRobotDataset
from lerobot.common.datasets.push_dataset_to_hub.aloha_hdf5_format import to_hf_dataset
from lerobot.common.datasets.push_dataset_to_hub.utils import concatenate_episodes, get_default_encoding
from lerobot.common.datasets.utils import calculate_episode_data_index, create_branch
from lerobot.common.datasets.video_utils import encode_video_frames
from lerobot.common.policies.factory import make_policy
from lerobot.common.robot_devices.robots.factory import make_robot
from lerobot.common.robot_devices.robots.utils import Robot
from lerobot.common.utils.utils import get_safe_torch_device, init_hydra_config, init_logging, set_global_seed
from lerobot.scripts.eval import get_pretrained_policy_path
from lerobot.scripts.push_dataset_to_hub import (
    push_dataset_card_to_hub,
    push_meta_data_to_hub,
    push_videos_to_hub,
    save_meta_data,
    push_dataset_to_hub,
)
from dataclasses import dataclass, field, replace
from lerobot.common.robot_devices.cameras.utils import Camera
import yaml
from lerobot.common.robot_devices.cameras.opencv import OpenCVCamera

logs = {}
def print_color(*args, color=None, attrs=(), **kwargs):
    import termcolor

    if len(args) > 0:
        args = tuple(termcolor.colored(arg, color=color, attrs=attrs) for arg in args)
    print(*args, **kwargs)
@cache
def is_headless():
    """Detects if python is running without a monitor."""
    try:
        import pynput  # noqa

        return False
    except Exception:
        print(
            "Error trying to import pynput. Switching to headless mode. "
            "As a result, the video stream from the cameras won't be shown, "
            "and you won't be able to change the control flow with keyboards. "
            "For more info, see traceback below.\n"
        )
        traceback.print_exc()
        print()
        return True
@dataclass

class camera_Config:
    cameras: dict[str, Camera] = field(default_factory=lambda: {})

@dataclass
class Args:
    agent: str = "none"
    robot_port: int = 6001
    wrist_camera_port: int = 5000
    base_camera_port: int = 5001
    hostname: str = "192.168.1.108"
    robot_type: str = None  # only needed for quest agent or spacemouse agent
    hz: int = 100
    start_joints: Optional[Tuple[float, ...]] = None

    gello_port: Optional[str] = None
    mock: bool = False
    use_save_interface: bool = False
    data_dir: str = "~/bc_data"
    bimanual: bool = False
    verbose: bool = False
def say(text, blocking=False):
    # Check if mac, linux, or windows.
    if platform.system() == "Darwin":
        cmd = f'say "{text}"'
    elif platform.system() == "Linux":
        cmd = f'spd-say "{text}"'
    elif platform.system() == "Windows":
        cmd = (
            'PowerShell -Command "Add-Type -AssemblyName System.Speech; '
            f"(New-Object System.Speech.Synthesis.SpeechSynthesizer).Speak('{text}')\""
        )

    if not blocking and platform.system() in ["Darwin", "Linux"]:
        # TODO(rcadene): Make it work for Windows
        # Use the ampersand to run command in the background
        cmd += " &"

    os.system(cmd)

def run_record(args,policy: torch.nn.Module | None = None):
    _, dataset_name = args.repo_id.split("/")
    num_episodes = args.num_episodes

    if dataset_name.startswith("eval_") and policy is None:
        raise ValueError(
            f"Your dataset name begins by 'eval_' ({dataset_name}) but no policy is provided ({policy})."
        )

    if not args.video:
        raise NotImplementedError()
    try:
        robot_client = ZMQClientRobot(port=args.robot_port, host=args.hostname)
        camera_clients = {}
        env = RobotEnv(robot_client, control_rate_hz=args.hz, camera_dict=camera_clients)
        usb_ports = glob.glob("/dev/serial/by-id/*")
        print(f"Found {len(usb_ports)} ports")
    except Exception as e:
        print(e)
        print("No robot found, please specify one or plug in robot")
        raise ValueError("No robot found, please specify one or plug in robot")
    if len(usb_ports) > 0:
        gello_port = usb_ports[0]
        print(f"using port {gello_port}")
    else:
        raise ValueError(
                "No gello port found, please specify one or plug in gello"
            )
    agent = GelloAgent(port=gello_port, start_joints=args.start_joints)
    curr_joints = env.get_obs()["joint_positions"]
    print(f"Current joints: {curr_joints}")
    print("curr_joints.shape",curr_joints.shape)
    start_pos = agent.act(env.get_obs())# 获取Gello的关节角,已经和机械臂配准过得关节角了，应该和机械臂的当前关节角一致
    obs = env.get_obs()#获取franka的状态信息
    joints = obs["joint_positions"]#提取状态信息中的关节角（包括夹爪）
    
    # 打开相机
    with open(args.camera_config, "r") as f:
        config_data = yaml.safe_load(f)

    camera_configs = {}
    for name, cam_conf in config_data.get("cameras", {}).items():
        cam_index = cam_conf["camera_index"]
        fps = cam_conf.get("fps")
        width = cam_conf.get("width")
        height = cam_conf.get("height")
        camera_configs[name] = OpenCVCamera(cam_index, fps=fps, width=width, height=height)
        camera_configs[name].connect()
        print(f"Connected to camera {cam_index}")
    cameras = camera_configs.copy()  
    # 相机  
    
    
    local_dir = Path(args.root) / args.repo_id
    if local_dir.exists() and args.force_override:
        shutil.rmtree(local_dir)

    episodes_dir = local_dir / "episodes"
    episodes_dir.mkdir(parents=True, exist_ok=True)

    videos_dir = local_dir / "videos"
    videos_dir.mkdir(parents=True, exist_ok=True)
    # Logic to resume data recording
    rec_info_path = episodes_dir / "data_recording_info.json"
    if rec_info_path.exists() and policy is None:
        with open(rec_info_path) as f:
            rec_info = json.load(f)
        episode_index = rec_info["last_episode_index"] + 1
    else:
        episode_index = 0

    if is_headless():
        logging.info(
            "Headless environment detected. On-screen cameras display and keyboard inputs will not be available."
        )
    exit_early = False
    rerecord_episode = False
    stop_recording = False
    listener =None
#     右箭头键（keyboard.Key.right）：设置exit_early为True，退出循环。
# 左箭头键（keyboard.Key.left）：设置rerecord_episode和exit_early为True，退出循环并重新记录最后一个剧集。
# Esc键（keyboard.Key.esc）：设置stop_recording和exit_early为True，停止数据记录并退出循环。
    if not is_headless():
        from pynput import keyboard
        def on_press(key):
            nonlocal exit_early, rerecord_episode, stop_recording
            try:
                if key == keyboard.Key.right:
                    print("Right arrow key pressed. Exiting loop...")
                    exit_early = True
                elif key == keyboard.Key.left:
                    print("Left arrow key pressed. Exiting loop and rerecord the last episode...")
                    rerecord_episode = True
                    exit_early = True
                elif key == keyboard.Key.esc:
                    print("Escape key pressed. Stopping data recording...")
                    stop_recording = True
                    exit_early = True
            except Exception as e:
                print(f"Error handling key press: {e}")

        listener = keyboard.Listener(on_press=on_press)
        listener.start()
    # if policy is not None:
    #     device = get_safe_torch_device(hydra_cfg.device, log=True)

    #     policy.eval()
    #     policy.to(device)

    #     torch.backends.cudnn.benchmark = True
    #     torch.backends.cuda.matmul.allow_tf32 = True
    #     set_global_seed(hydra_cfg.seed)

    #     # override fps using policy fps
    #     fps = hydra_cfg.env.fps
    # 固定值30
    fps = fps if fps is not None else 30
    timestamp = 0
    start_warmup_t = time.perf_counter()
    is_warmup_print = False
    while timestamp < args.warmup_time_s:
 
        if not is_warmup_print:
            logging.info("Warming up (no data recording)")
            # say("Warming up")
            is_warmup_print = True

        start_loop_t = time.perf_counter()

        if policy is None:
            observation, action = teleop_step(env, agent, cameras, record_data=True)
        else:
            # observation = capture_observation()
            raise NotImplementedError
        dt_s = time.perf_counter() - start_loop_t
        busy_wait(1 / fps - dt_s)

        dt_s = time.perf_counter() - start_loop_t
        log_control_info(logs,dt_s, fps=fps)

        timestamp = time.perf_counter() - start_warmup_t


    print("****************warmup done**********************")
    # say("warmup done")
    futures = []
    image_keys = []
    not_image_keys = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.num_image_writers) as executor:
        # Start recording all episodes
        print("episode_index:",episode_index)
        print("num_episodes:",args.num_episodes)
        while episode_index < args.num_episodes:
            logging.info(f"Recording episode {episode_index}")
            # say(f"Recording episode {episode_index}")
            # input("按回车开始采集...")

            ep_dict = {}
            frame_index = 0
            timestamp = 0
            start_episode_t = time.perf_counter()
            while timestamp < args.episode_time_s:
                start_loop_t = time.perf_counter()

                if policy is None:
                    observation, action = teleop_step(env, agent, cameras, record_data=True)
                else:
                    raise  NotImplementedError

                image_keys = [key for key in observation if "image" in key]
                not_image_keys = [key for key in observation if "image" not in key]

                for key in image_keys:
                    futures += [
                        executor.submit(
                            save_image, observation[key], key, frame_index, episode_index, videos_dir
                        )
                    ]
                for key in not_image_keys:
                    if key not in ep_dict:
                        ep_dict[key] = []
                    ep_dict[key].append(observation[key])
                for key in action:
                    if key not in ep_dict:
                        ep_dict[key] = []
                    ep_dict[key].append(action[key])
                frame_index += 1

                dt_s = time.perf_counter() - start_loop_t
                busy_wait(1 / fps - dt_s)

                dt_s = time.perf_counter() - start_loop_t
                log_control_info(logs, dt_s, fps=fps)

                timestamp = time.perf_counter() - start_episode_t
                if exit_early:
                    exit_early = False
                    break
            if not stop_recording:
                # Start resetting env while the executor are finishing
                logging.info("Reset the environment")
                # say("Reset the environment")

            timestamp = 0
            start_vencod_t = time.perf_counter()

            # During env reset we save the data and encode the videos
            num_frames = frame_index
            for key in image_keys:
                tmp_imgs_dir = videos_dir / f"{key}_episode_{episode_index:06d}"
                fname = f"{key}_episode_{episode_index:06d}.mp4"
                video_path = local_dir / "videos" / fname
                if video_path.exists():
                    video_path.unlink()
                # Store the reference to the video frame, even tho the videos are not yet encoded
                ep_dict[key] = []
                for i in range(num_frames):
                    ep_dict[key].append({"path": f"videos/{fname}", "timestamp": i / fps})

            for key in not_image_keys:
                ep_dict[key] = torch.stack(ep_dict[key])

            for key in action:
                ep_dict[key] = torch.stack(ep_dict[key])

            ep_dict["episode_index"] = torch.tensor([episode_index] * num_frames)
            ep_dict["frame_index"] = torch.arange(0, num_frames, 1)
            ep_dict["timestamp"] = torch.arange(0, num_frames, 1) / fps

            done = torch.zeros(num_frames, dtype=torch.bool)
            done[-1] = True
            ep_dict["next.done"] = done

            ep_path = episodes_dir / f"episode_{episode_index}.pth"
            
            torch.save(ep_dict, ep_path)

            rec_info = {
                "last_episode_index": episode_index,
            }
            with open(rec_info_path, "w") as f:
                json.dump(rec_info, f)

            is_last_episode = stop_recording or (episode_index == (num_episodes - 1))
            if policy is None:
                print(f"保存第{episode_index}帧数据完成。一轮采集数据结束，暂停恢复环境，leader机械臂返回原点。")
                input("按回车，开始新一轮恢复采集")
            
            # Skip updating episode index which forces re-recording episode
            if rerecord_episode:
                rerecord_episode = False
                continue

            episode_index += 1

            if is_last_episode:
                logging.info("Done recording")
                # say("Done recording", blocking=True)
                # if not is_headless():
                #     listener.stop()

                logging.info("Waiting for threads writing the images on disk to terminate...")
                for _ in tqdm.tqdm(
                    concurrent.futures.as_completed(futures), total=len(futures), desc="Writting images"
                ):
                    pass
                break
    
    for name in cameras:
        cameras[name].disconnect()
    num_episodes = episode_index

    logging.info("Encoding videos")
    # say("Encoding videos")
    # Use ffmpeg to convert frames stored as png into mp4 videos
    for episode_index in tqdm.tqdm(range(num_episodes)):
        for key in image_keys:
            tmp_imgs_dir = videos_dir / f"{key}_episode_{episode_index:06d}"
            fname = f"{key}_episode_{episode_index:06d}.mp4"
            video_path = local_dir / "videos" / fname
            if video_path.exists():
                # Skip if video is already encoded. Could be the case when resuming data recording.
                continue
            # note: `encode_video_frames` is a blocking call. Making it asynchronous shouldn't speedup encoding,
            # since video encoding with ffmpeg is already using multithreading.
            encode_video_frames(tmp_imgs_dir, video_path, fps, overwrite=True)
            shutil.rmtree(tmp_imgs_dir)
    logging.info("Concatenating episodes")
    ep_dicts = []
    for episode_index in tqdm.tqdm(range(num_episodes)):
        ep_path = episodes_dir / f"episode_{episode_index}.pth"
        ep_dict = torch.load(ep_path)
        ep_dicts.append(ep_dict)
    data_dict = concatenate_episodes(ep_dicts)

    total_frames = data_dict["frame_index"].shape[0]
    data_dict["index"] = torch.arange(0, total_frames, 1)

   

    hf_dataset = to_hf_dataset(data_dict, args.video)
    episode_data_index = calculate_episode_data_index(hf_dataset)
    info = {
        "codebase_version": CODEBASE_VERSION,
        "fps": fps,
        "video": args.video,
    }
    if args.video:
        info["encoding"] = get_default_encoding()

    lerobot_dataset = LeRobotDataset.from_preloaded(
        repo_id=args.repo_id,
        hf_dataset=hf_dataset,
        episode_data_index=episode_data_index,
        info=info,
        videos_dir=videos_dir,
    )
    if args.run_compute_stats:
        logging.info("Computing dataset statistics")
        say("Computing dataset statistics")
        stats = compute_stats(lerobot_dataset)
        lerobot_dataset.stats = stats
    else:
        stats = {}
        logging.info("Skipping computation of the dataset statistics")

    hf_dataset = hf_dataset.with_format(None)  # to remove transforms that cant be saved
    hf_dataset.save_to_disk(str(local_dir / "train"))

    meta_data_dir = local_dir / "meta_data"
    save_meta_data(info, stats, episode_data_index, meta_data_dir)  
#     push_dataset_to_hub(
#     raw_dir=str(local_dir / "train"),
#     raw_format="gello",
#     repo_id=args.repo_id,
#     push_to_hub=True,
#     local_dir=Path("data/lerobot/debug/hub/"),
#     fps=fps,
#     video=args.video,
#     force_override=args.force_override,
# )
    from datasets import load_from_disk

    hf_dataset_path = local_dir / "train"
    dataset = load_from_disk(str(hf_dataset_path))
    dataset.push_to_hub(repo_id=args.repo_id, private=True)  # 或 False
    
    logging.info("Exiting")
    say("Exiting")
    return lerobot_dataset
def to_python_type(obj):
    if isinstance(obj, torch.Tensor):
        if obj.ndim == 0:
            return obj.item()
        return obj.tolist()
    elif isinstance(obj, np.ndarray):
        if obj.ndim == 0:
            return obj.item()
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: to_python_type(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [to_python_type(x) for x in obj]
    else:
        return obj
def save_image(img_tensor, key, frame_index, episode_index, videos_dir):
    """
    将输入的张量图像保存到指定路径下。
    
    Args:
        img_tensor (Tensor): 输入的张量图像，形状为 (H, W, C)，数据类型为 float32 或 uint8。
        key (str): 图像保存路径的标识字符串。
        frame_index (int): 当前帧的索引号。
        episode_index (int): 当前片段的索引号。
        videos_dir (Path): 图像保存路径的根目录。
    
    Returns:
        None
    
    """
    img = Image.fromarray(img_tensor.numpy())
    path = videos_dir / f"{key}_episode_{episode_index:06d}" / f"frame_{frame_index:06d}.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(str(path), quality=100)
def log_control_info(logs,dt_s, episode_index=None, frame_index=None, fps=None):
    log_items = []
    if episode_index is not None:
        log_items += [f"ep:{episode_index}"]
    if frame_index is not None:
        log_items += [f"frame:{frame_index}"]

    def log_dt(shortname, dt_val_s):
        log_items.append(f"{shortname}:{dt_val_s * 1000:5.2f} ({1 / dt_val_s:3.1f}hz)")

    log_dt("dt", dt_s)

    for key in logs:
        if key.endswith("_dt_s"):
            shortname = key.replace("_dt_s", "")
            log_dt(shortname, logs[key])

    info_str = " ".join(log_items)
    if fps is not None and (1 / dt_s < fps - 1):
        info_str = colored(info_str, "yellow")

    # logging.info(info_str)


def busy_wait(seconds):
    # Significantly more accurate than `time.sleep`, and mendatory for our use case,
    # but it consumes CPU cycles.
    # TODO(rcadene): find an alternative: from python 11, time.sleep is precise
    end_time = time.perf_counter() + seconds
    while time.perf_counter() < end_time:
        pass
def teleop_step(env, agent, cameras, record_data=False):
    

# ========== Leader/Follower Joint State ==========
    before_joint_t = time.perf_counter()
    leader_pos = {}
    obs = env.get_obs()
    
    action = agent.act(obs)
    after_joint_t = time.perf_counter()
    logs["read_obs_dt_s"] = after_joint_t - before_joint_t
    before_ctrl_t = time.perf_counter()
    obs = env.step(action)
    after_ctrl_t = time.perf_counter()
    logs["step_action_dt_s"] = after_ctrl_t - before_ctrl_t
    if not record_data:
        return

    obs = env.get_obs()
    state = []
    follower_pos = obs["joint_positions"]
    state.append(follower_pos.copy())  # 推荐，用 copy 避免共享引用
    images={}
    for name in cameras:
        before_camread_t = time.perf_counter()
        images[name] = cameras[name].async_read()
        after_cam_t = time.perf_counter()
        logs[f"read_camera_{name}_dt_s"] = after_cam_t - before_camread_t
    obs_dict, action_dict = {}, {}
    obs_dict["observation.state"] = torch.from_numpy(np.array(state))
    action_dict["action"] = torch.from_numpy(action)
    for name in cameras:
        obs_dict[f"observation.images.{name}"] = torch.from_numpy(images[name])

    step_end_t = time.perf_counter()
    step_dt_s = step_end_t - before_joint_t
    logs["step_dt_s"] = step_dt_s


    return obs_dict, action_dict

            




 
    

    





def run_teleoperate(args):
    if args.agent == "gello":
        try:
            robot_client = ZMQClientRobot(port=args.robot_port, host=args.hostname)
            camera_clients = {}
            env = RobotEnv(robot_client, control_rate_hz=args.hz, camera_dict=camera_clients)
            usb_ports = glob.glob("/dev/serial/by-id/*")
            print(f"Found {len(usb_ports)} ports")
        except Exception as e:
            print(e)
            print("No robot found, please specify one or plug in robot")
            raise ValueError("No robot found, please specify one or plug in robot")
        if len(usb_ports) > 0:
            gello_port = usb_ports[0]
            print(f"using port {gello_port}")
        else:
            raise ValueError(
                "No gello port found, please specify one or plug in gello"
            )
        agent = GelloAgent(port=gello_port, start_joints=args.start_joints)
        curr_joints = env.get_obs()["joint_positions"]
        print(f"Current joints: {curr_joints}")
        print("curr_joints.shape",curr_joints.shape)
        start_pos = agent.act(env.get_obs())# 获取Gello的关节角,已经和机械臂配准过得关节角了，应该和机械臂的当前关节角一致
        obs = env.get_obs()#获取franka的状态信息
        joints = obs["joint_positions"]#提取状态信息中的关节角（包括夹爪）
        abs_deltas=np.abs(start_pos - joints)
        abs_deltas[-1]=0# 夹爪的关节角不需要考虑
        print("Start pos: ", start_pos)
        print("Joints: ", joints)
        print("Deltas: ", abs_deltas)

        id_max_joint_delta = np.argmax(abs_deltas)
        max_joint_delta = 0.8
        if abs_deltas[id_max_joint_delta] > max_joint_delta:
            id_mask = abs_deltas > max_joint_delta
            print()
            ids = np.arange(len(id_mask))[id_mask]
            for i, delta, joint, current_j in zip(
                ids,
                abs_deltas[id_mask],
                start_pos[id_mask],
                joints[id_mask],
            ):
                print(
                    f"joint[{i}]: \t delta: {delta:4.3f} , leader: \t{joint:4.3f} , follower: \t{current_j:4.3f}"
                )
            return
        print(f"Start pos: {len(start_pos)}", f"Joints: {len(joints)}")
        assert len(start_pos) == len(
        joints
        ), f"agent output dim = {len(start_pos)}, but env dim = {len(joints)}"
        # 在控制开始前，让机械臂从当前状态慢慢过渡到 leader 当前的位置，防止一下子跳过去。
        max_delta = 0.05
        for _ in range(25):
            obs = env.get_obs()
            command_joints = agent.act(obs)
            current_joints = obs["joint_positions"]
            delta = command_joints - current_joints
            max_joint_delta = np.abs(delta).max()
            if max_joint_delta > max_delta:
                delta = delta / max_joint_delta * max_delta
            env.step(current_joints + delta)
        obs = env.get_obs()
        joints = obs["joint_positions"]
        action = agent.act(obs)
        if (action - joints > 0.5).any():
            print("Action is too big")

            # print which joints are too big
            joint_index = np.where(action - joints > 0.8)
            for j in joint_index:
                print(
                    f"Joint [{j}], leader: {action[j]}, follower: {joints[j]}, diff: {action[j] - joints[j]}"
                )
            exit()
        print_color("\nStart 🚀🚀🚀", color="green", attrs=("bold",))
        save_path = None
        start_time = time.time()
        while True:
            num = time.time() - start_time
            message = f"\rTime passed: {round(num, 2)}          "
            print_color(
                message,
                color="white",
                attrs=("bold",),
                end="",
                flush=True,
            )
            action = agent.act(obs)
            dt = datetime.datetime.now()
            
            obs = env.step(action)
    else:
        raise ValueError(f"Unknown agent: {args.agent}")
    
                       

def main(args):
    # 单臂运行
    if args.agent == "gello":
        # robot_client = ZMQClientRobot(port=args.robot_port, host=args.hostname)
        # camera_clients = {}
        # 用try，如果超时，则退出
        try:
            robot_client = ZMQClientRobot(port=args.robot_port, host=args.hostname)
            camera_clients = {}
            env = RobotEnv(robot_client, control_rate_hz=args.hz, camera_dict=camera_clients)
            usb_ports = glob.glob("/dev/serial/by-id/*")
            print(f"Found {len(usb_ports)} ports")
        except Exception as e:
            print(e)
            print("No robot found, please specify one or plug in robot")
            raise ValueError("No robot found, please specify one or plug in robot")
        # env = RobotEnv(robot_client, control_rate_hz=args.hz, camera_dict=camera_clients)
        # usb_ports = glob.glob("/dev/serial/by-id/*")
        # print(f"Found {len(usb_ports)} ports")
        if len(usb_ports) > 0:
            gello_port = usb_ports[0]
            print(f"using port {gello_port}")
        else:
            raise ValueError(
                "No gello port found, please specify one or plug in gello"
            )
        agent = GelloAgent(port=gello_port, start_joints=args.start_joints)
        curr_joints = env.get_obs()["joint_positions"]
        print(f"Current joints: {curr_joints}")
        print("curr_joints.shape",curr_joints.shape)
        start_pos = agent.act(env.get_obs())# 获取Gello的关节角,已经和机械臂配准过得关节角了，应该和机械臂的当前关节角一致
        obs = env.get_obs()#获取franka的状态信息
        joints = obs["joint_positions"]#提取状态信息中的关节角（包括夹爪）
        abs_deltas=np.abs(start_pos - joints)
        abs_deltas[-1]=0# 夹爪的关节角不需要考虑
        print("Start pos: ", start_pos)
        print("Joints: ", joints)
        print("Deltas: ", abs_deltas)

        id_max_joint_delta = np.argmax(abs_deltas)
        max_joint_delta = 0.8
        if abs_deltas[id_max_joint_delta] > max_joint_delta:
            id_mask = abs_deltas > max_joint_delta
            print()
            ids = np.arange(len(id_mask))[id_mask]
            for i, delta, joint, current_j in zip(
                ids,
                abs_deltas[id_mask],
                start_pos[id_mask],
                joints[id_mask],
            ):
                print(
                    f"joint[{i}]: \t delta: {delta:4.3f} , leader: \t{joint:4.3f} , follower: \t{current_j:4.3f}"
                )
            return
        print(f"Start pos: {len(start_pos)}", f"Joints: {len(joints)}")
        assert len(start_pos) == len(
        joints
        ), f"agent output dim = {len(start_pos)}, but env dim = {len(joints)}"
        # 在控制开始前，让机械臂从当前状态慢慢过渡到 leader 当前的位置，防止一下子跳过去。
        max_delta = 0.05
        for _ in range(25):
            obs = env.get_obs()
            command_joints = agent.act(obs)
            current_joints = obs["joint_positions"]
            delta = command_joints - current_joints
            max_joint_delta = np.abs(delta).max()
            if max_joint_delta > max_delta:
                delta = delta / max_joint_delta * max_delta
            env.step(current_joints + delta)
        obs = env.get_obs()
        joints = obs["joint_positions"]
        action = agent.act(obs)
        if (action - joints > 0.5).any():
            print("Action is too big")

            # print which joints are too big
            joint_index = np.where(action - joints > 0.8)
            for j in joint_index:
                print(
                    f"Joint [{j}], leader: {action[j]}, follower: {joints[j]}, diff: {action[j] - joints[j]}"
                )
            exit()
        print_color("\nStart 🚀🚀🚀", color="green", attrs=("bold",))
        save_path = None
        start_time = time.time()
        while True:
            num = time.time() - start_time
            message = f"\rTime passed: {round(num, 2)}          "
            print_color(
                message,
                color="white",
                attrs=("bold",),
                end="",
                flush=True,
            )
            action = agent.act(obs)
            dt = datetime.datetime.now()
            
            obs = env.step(action)
        
      









    


def none_or_int(value):
    if value == "None":
        return None
    return int(value)

if __name__ == "__main__":
    # main(tyro.cli(Args))
    parser=argparse.ArgumentParser()
    subparsers=parser.add_subparsers(dest="mode")
    base_parser = argparse.ArgumentParser(add_help=False)

    # base_parser.add_argument("--fps", type=int, default=30, help="Frames per second")
    base_parser.add_argument("--robot-port", type=int, default=6001)
    base_parser.add_argument("--hostname", type=str, default="192.168.1.108")
    # RECORD 模式
    parser_record = subparsers.add_parser("record", parents=[base_parser])
    # parser_record.add_argument("--agent", default="gello")
   
    # parser_record.add_argument("--robot-port", type=int, default=6001)
    # parser_record.add_argument("--hostname", default="192.168.1.108")
    parser_record.add_argument(
        "--fps", type=none_or_int, default=None, help="Frames per second (set to None to disable)"
    )
    parser_record.add_argument("--root", type=str, default="data")
    parser_record.add_argument("--repo-id", type=str, default="SunJincheng/gello")
    
    # parser_record.add_argument("--num-episodes", type=int, default=50)
    parser_record.add_argument("--warmup-time-s", type=int, default=10)
    parser_record.add_argument("--episode-time-s", type=int, default=60)
    parser_record.add_argument("--reset-time-s", type=int, default=10)
    parser_record.add_argument("--num-episodes", type=int, default=50, help="Number of episodes to record.")
    parser_record.add_argument(
    "--hz",
    type=float,
    default=100.0,
    help="Control loop frequency in Hz (default: 30)"
    )

    parser_record.add_argument(
        "--run-compute-stats",
        type=int,
        default=1,
        help="By default, run the computation of the data statistics at the end of data collection. Compute intensive and not required to just replay an episode.",
    )
    parser_record.add_argument(
        "--push-to-hub",
        type=int,
        default=1,
        help="Upload dataset to Hugging Face hub.",
    )
    parser_record.add_argument(
        "--tags",
        type=str,
        nargs="*",
        help="Add tags to your dataset on the hub.",
    )
    parser_record.add_argument(
        "--num-image-writers",
        type=int,
        default=8,
        help="Number of threads writing the frames as png images on disk. Don't set too much as you might get unstable fps due to main thread being blocked.",
    )
    parser_record.add_argument(
        "--force-override",
        type=int,
        default=0,
        help="By default, data recording is resumed. When set to 1, delete the local directory and start data recording from scratch.",
    )
    parser_record.add_argument(
        "-p",
        "--pretrained-policy-name-or-path",
        type=str,
        help=(
            "Either the repo ID of a model hosted on the Hub or a path to a directory containing weights "
            "saved using `Policy.save_pretrained`."
        ),
    )
    parser_record.add_argument(
        "--policy-overrides",
        type=str,
        nargs="*",
        help="Any key=value arguments to override config values (use dots for.nested=overrides)",
    )
    parser_record.add_argument(
    "--camera-config",
    type=Path,
    default=Path("lerobot/configs/robot/camera.yaml"),
    help="Path to a YAML file containing camera configuration for one or more cameras. "
        "Defaults to lerobot/configs/robot/camera.yaml",
)
    

    # video=True,root="data",
    parser_record.add_argument(
        "--video",
        default=True,
        help="Whether to record video.",
    )
    parser_record.add_argument(
    "--start_joints",
    type=float,
    nargs="+",
    default=None,
    help="用于设置初始关节角度的列表，例如: --start 0.1 -0.2 0.3 1.0 0.0 -0.5 0.8。必须与机器人关节数一致（例如7个值）。"
    )
    
    # TELEOPERATE 模式
    # python control_robot.py teleoperate --fps 30 --agent gello --robot-port 6001 --hostname 192.168.1.108 --use-save-interface

    parser_teleop = subparsers.add_parser("teleoperate", parents=[base_parser])
    parser_teleop.add_argument("--agent", default="gello")
    # parser_teleop.add_argument("--robot-port", type=int, default=6001)
    # parser_teleop.add_argument("--hostname", default="192.168.1.108")
    parser_teleop.add_argument("--hz", type=int, default=100)
    # start_joints: Optional[Tuple[float, ...]] = None
   
    parser_teleop.add_argument(
    "--start_joints",
    type=float,
    nargs="+",
    default=None,
    help="用于设置初始关节角度的列表，例如: --start 0.1 -0.2 0.3 1.0 0.0 -0.5 0.8。必须与机器人关节数一致（例如7个值）。"
)
   

    # REPLAY 模式
    parser_replay = subparsers.add_parser("replay", parents=[base_parser])
    parser_replay.add_argument("--root", type=str, required=True)
    parser_replay.add_argument("--repo-id", type=str, required=True)
    parser_replay.add_argument("--episode", type=int, default=0)

    args = parser.parse_args()
    kwargs = vars(args)
    init_logging()
    if args.mode == "record":
        print_color("📹 Starting RECORD mode", color="cyan", attrs=["bold"])
        pretrained_policy_name_or_path = args.pretrained_policy_name_or_path
        policy_overrides = args.policy_overrides
        del kwargs["pretrained_policy_name_or_path"]
        del kwargs["policy_overrides"]

        policy_cfg = None
        if pretrained_policy_name_or_path is not None:
            pretrained_policy_path = get_pretrained_policy_path(pretrained_policy_name_or_path)
            policy_cfg = init_hydra_config(pretrained_policy_path / "config.yaml", policy_overrides)
            policy = make_policy(hydra_cfg=policy_cfg, pretrained_policy_name_or_path=pretrained_policy_path)
            run_record( policy, policy_cfg, **kwargs)
        else:
            print("Starting record")
            kwargs.pop("mode", None)  # 删除 mode 参数，如果不存在则无事发生
            run_record( args)
        # TODO: 填充 record 数据采集逻辑
        # 例如：run_record(args)
    elif args.mode == "teleoperate":
        print_color("🕹️ Starting TELEOPERATION mode", color="yellow", attrs=["bold"])
        # TODO: 填充 teleoperate 逻辑
        run_teleoperate(args)
        # 例如：run_teleop(args)
    elif args.mode == "replay":
        print_color("🎬 Starting REPLAY mode", color="magenta", attrs=["bold"])
        # TODO: 填充 replay 逻辑
        # 例如：run_replay(args)
    else:
        raise ValueError(f"Unknown mode: {args.mode}")




