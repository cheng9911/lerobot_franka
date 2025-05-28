# main.py
import argparse
import yaml
from pathlib import Path
from lerobot.common.robot_devices.cameras.opencv import OpenCVCamera


def load_camera_configs(config_path):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config["cameras"]

def build_cameras(camera_dict):
    cameras = []
    for name, cfg in camera_dict.items():
        camera = OpenCVCamera(cfg["camera_index"],
                              fps=cfg.get("fps"),
                              width=cfg.get("width"),
                              height=cfg.get("height"))
        camera.connect()
        cameras.append(camera)
    return cameras

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--images-dir", type=Path, default=Path("outputs/captured_images"))
    parser.add_argument("--record-time-s", type=float, default=10)
    args = parser.parse_args()

    camera_dict = load_camera_configs(args.config)
    cameras = build_cameras(camera_dict)
    

    

    for cam in cameras:
        cam.disconnect()
