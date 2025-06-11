from dataclasses import dataclass, field, replace
from lerobot.common.robot_devices.cameras.utils import Camera
import yaml
from lerobot.common.robot_devices.cameras.opencv import OpenCVCamera
import argparse
from pathlib import Path
import time
import traceback
from functools import cache
import json
import logging
from PIL import Image
import concurrent.futures
import torch
import tqdm
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

def record_images(args):
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
    output_dir = Path(args.output_dir)
    local_dir = Path(args.root) 
    rec_info_path = local_dir / "data_recording_info.json"
    if rec_info_path.exists() :
        with open(rec_info_path) as f:
            rec_info = json.load(f)
        episode_index = rec_info["last_episode_index"] + 1
    else:
        episode_index = 0
    #     右箭头键（keyboard.Key.right）：设置exit_early为True，退出循环。
# 左箭头键（keyboard.Key.left）：设置rerecord_episode和exit_early为True，退出循环并重新记录最后一个剧集。
# Esc键（keyboard.Key.esc）：设置stop_recording和exit_early为True，停止数据记录并退出循环。
    exit_early = False
    rerecord_episode = False
    stop_recording = False
    listener =None
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
        fps = fps if fps is not None else 30
        timestamp = 0
        start_warmup_t = time.perf_counter()
        is_warmup_print = False
        while timestamp < args.warmup_time_s:
            start_loop_t = time.perf_counter()
            images={}
            for name in cameras:
                
                images[name] = cameras[name].async_read()
        
            dt_s = time.perf_counter() - start_loop_t
            busy_wait(1 / fps - dt_s)

            dt_s = time.perf_counter() - start_loop_t

            timestamp = time.perf_counter() - start_warmup_t
        
    print("****************warmup done**********************")
    futures = []
    image_keys = []
    not_image_keys = []
    print("episode_index:",episode_index)
    print("num_episodes:",args.num_episodes)
    num_episodes = args.num_episodes
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.num_image_writers) as executor:
        while episode_index < args.num_episodes:
            logging.info(f"Recording episode {episode_index}")
            observation = teleop_step(cameras)
            image_keys = [key for key in observation if "image" in key]
            not_image_keys = [key for key in observation if "image" not in key]
            
            

            for key in image_keys:
                # 取key的最后一个字符串
                name = key.split(".")[-1]
               
                futures += [
                    executor.submit(
                        save_image, observation[key],  episode_index, output_dir,name               )
                ]
            dt_s = time.perf_counter() - start_loop_t
            busy_wait(1 / fps - dt_s)

            dt_s = time.perf_counter() - start_loop_t
            if exit_early:
                exit_early = False
                break
            rec_info = {
                "last_episode_index": episode_index,
            }
            with open(rec_info_path, "w") as f:
                json.dump(rec_info, f)
            is_last_episode = stop_recording or (episode_index == (num_episodes - 1))
            print(f"保存第{episode_index}帧数据完成。一轮采集数据结束，暂停恢复环境，leader机械臂返回原点。")
            input("按回车，开始新一轮恢复采集")
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

def teleop_step(cameras):   
    images={}
    for name in cameras:
        before_camread_t = time.perf_counter()
        images[name] = cameras[name].async_read()
        after_cam_t = time.perf_counter()
        
    obs_dict = {}
    for name in cameras:
        obs_dict[f"observation.images.{name}"] = torch.from_numpy(images[name])
    return obs_dict
def save_image(img_tensor,  episode_index, videos_dir,name):
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
    # 存成0001_head.png或者0001_flan.png
    path = videos_dir /  f"{episode_index:04d}_{name}.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(str(path), quality=100)
def busy_wait(seconds):
    # Significantly more accurate than `time.sleep`, and mendatory for our use case,
    # but it consumes CPU cycles.
    # TODO(rcadene): find an alternative: from python 11, time.sleep is precise
    end_time = time.perf_counter() + seconds
    while time.perf_counter() < end_time:
        pass

    
if __name__ == "__main__":
    parser=argparse.ArgumentParser()
    subparsers=parser.add_subparsers(dest="mode")
    base_parser = argparse.ArgumentParser(add_help=False)
    parser_record = subparsers.add_parser("record", parents=[base_parser])
    parser_record.add_argument(
    "--camera-config",
    type=Path,
    default=Path("lerobot/configs/robot/camera_two.yaml"),
    help="Path to a YAML file containing camera configuration for one or more cameras. "
        "Defaults to lerobot/configs/robot/camera.yaml",
)
    parser_record.add_argument(
    "--output-dir",
    type=Path,
    default=Path("lerobot/resnet/data/images"),
    help="Path to a YAML file containing camera configuration for one or more cameras. "
        "Defaults to lerobot/configs/robot/camera.yaml",
)
    parser_record.add_argument("--warmup-time-s", type=int, default=10)
    parser_record.add_argument("--num-episodes", type=int, default=200, help="Number of episodes to record.")
    parser_record.add_argument("--root", type=str, default="lerobot/resnet/data")

    parser_record.add_argument(
        "--num-image-writers",
        type=int,
        default=8,
        help="Number of threads writing the frames as png images on disk. Don't set too much as you might get unstable fps due to main thread being blocked.",
    )
    args = parser.parse_args()
    kwargs = vars(args)

    if args.mode == "record":
        record_images(args)