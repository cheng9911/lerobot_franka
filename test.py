# from huggingface_hub import HfApi
# import os
# api = HfApi(token=os.getenv("HF_TOKEN"))
# api.upload_folder(
#     folder_path="/home/rocos/Documents/GitHub/lerobot_franka/data/SunJincheng/gello",
#     repo_id="SunJincheng/gello",
#     repo_type="dataset",
# )
import os

# 设置代理
os.environ['HTTP_PROXY'] = 'http://127.0.0.1:7890'
os.environ['HTTPS_PROXY'] = 'http://127.0.0.1:7890'

from huggingface_hub import HfApi

# 使用 token 认证
api = HfApi(token=os.getenv("HF_TOKEN"))

# 上传数据集
api.upload_folder(
    folder_path="/home/rocos/Documents/GitHub/lerobot_franka/data/SunJincheng/panda",
    repo_id="SunJincheng/panda",
    repo_type="dataset",
)
