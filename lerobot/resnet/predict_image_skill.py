import argparse
import yaml
import time
import torch
import gc
import json
from pathlib import Path
from PIL import Image

from lerobot.common.robot_devices.cameras.opencv import OpenCVCamera
from clip_train_text import ClipSkillPredictor, extract_features_batch, preprocess, DEVICE


def busy_wait(seconds):
    end_time = time.perf_counter() + seconds
    while time.perf_counter() < end_time:
        pass


def teleop_step(cameras):
    images = {}
    for name in cameras:
        images[name] = cameras[name].async_read()
    obs_dict = {}
    for name in cameras:
        obs_dict[f"observation.images.{name}"] = torch.from_numpy(images[name])
    return obs_dict


def save_predictions(predictions, save_path):
    save_path.parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, "w") as f:
        json.dump(predictions, f, indent=2)
    print(f"已保存预测结果到 {save_path}")


def predict_images(args):
    # 加载相机配置
    with open(args.camera_config, "r") as f:
        config_data = yaml.safe_load(f)

    camera_configs = {}
    for name, cam_conf in config_data.get("cameras", {}).items():
        cam_index = cam_conf["camera_index"]
        fps = cam_conf.get("fps", 30)
        width = cam_conf.get("width")
        height = cam_conf.get("height")
        camera = OpenCVCamera(cam_index, fps=fps, width=width, height=height)
        camera.connect()
        camera_configs[name] = camera
        print(f"Connected to camera {name} (index {cam_index})")

    # 预热
    print("Warming up cameras...")
    start_time = time.perf_counter()
    while time.perf_counter() - start_time < args.warmup_time_s:
        for cam in camera_configs.values():
            _ = cam.async_read()
        busy_wait(1.0 / fps)
    print("************ Warmup done ************")

    # 加载模型
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()
    model = ClipSkillPredictor().to(DEVICE)
    model.load_state_dict(torch.load(args.model_path, map_location=DEVICE))
    model.eval()
    print("Model loaded.")

    # 预测并记录
    predictions = []
    for i in range(args.num_predicts):
        print(f"\n====== 第 {i+1} 次预测 ======")
        obs = teleop_step(camera_configs)

        try:
            head_tensor = obs["observation.images.head"]
            flan_tensor = obs["observation.images.flan"]

            head_img = preprocess(Image.fromarray(head_tensor.numpy()).convert("RGB")).unsqueeze(0).to(DEVICE)
            flan_img = preprocess(Image.fromarray(flan_tensor.numpy()).convert("RGB")).unsqueeze(0).to(DEVICE)

            with torch.no_grad():
                vision_feat = extract_features_batch(head_img, flan_img)
                dummy_text_feat = torch.zeros((1, 512), device=DEVICE)
                fused_feat = torch.cat([vision_feat, dummy_text_feat], dim=-1)
                logits = model(fused_feat)
                pred = torch.argmax(logits, dim=-1).squeeze(0).tolist()

            print(f"预测技能序列: {pred}")
            predictions.append({
                # "episode_id": i,
                "skill_sequence": pred
            })

        except Exception as e:
            print(f"预测失败: {e}")

        if i < args.num_predicts - 1:
            input("按下回车开始下一轮预测...")

    for cam in camera_configs.values():
        cam.disconnect()

    save_path = Path("lerobot/resnet/data/llm/predicted_skills.json")
    save_predictions(predictions, save_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--camera-config",
        type=Path,
        default=Path("lerobot/configs/robot/camera_two.yaml"),
        help="YAML 文件，定义 head/flan 相机参数。",
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=Path("lerobot/resnet/clip_skill_model.pt"),
        help="已训练模型的路径（.pt 文件）",
    )
    parser.add_argument("--warmup-time-s", type=int, default=5, help="相机预热时间（秒）")
    parser.add_argument("--num-predicts", type=int, default=5, help="进行预测的次数")

    args = parser.parse_args()
    predict_images(args)
