# streamlit run lerobot/resnet/task_think.py --server.address=192.168.1.14
import streamlit as st
import json
import subprocess
import platform
import os
from openai import OpenAI
import asyncio
import edge_tts
import threading
import tempfile
from gtts import gTTS
import pyttsx3


def say(text, blocking=False):
    system = platform.system()

    def play_audio(mp3_path):
        if system == "Darwin":
            # macOS 用 afplay 播放
            subprocess.run(['afplay', mp3_path])
        elif system == "Linux":
            # Linux 用 mpg123 播放
            subprocess.run(['mpg123', '-q', mp3_path])
        elif system == "Windows":
            # Windows 用 powershell 播放
            ps_command = f'(New-Object Media.SoundPlayer "{mp3_path}").PlaySync();'
            subprocess.run(['powershell', '-Command', ps_command], shell=True)
        else:
            print("Unsupported OS for audio playback.")

    def run():
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
            mp3_path = f.name
        try:
            tts = gTTS(text=text, lang='zh')
            tts.save(mp3_path)
            play_audio(mp3_path)
        finally:
            if os.path.exists(mp3_path):
                os.remove(mp3_path)

    if blocking:
        run()
    else:
        threading.Thread(target=run, daemon=True).start()
print("开始任务")
# say("开始任务", blocking=True)


# 用于语音反馈（可选）
# def say(text, blocking=False):
#     system = platform.system()
#     if system == "Darwin":
#         cmd = ['say', text]
#     elif system == "Linux":
#         cmd = ['espeak-ng', '-v', 'cmn-latn-pinyin', text]
#     elif system == "Windows":
#         ps_command = (
#             f'Add-Type -AssemblyName System.Speech;'
#             f'(New-Object System.Speech.Synthesis.SpeechSynthesizer).Speak("{text}")'
#         )
#         cmd = ['powershell', '-Command', ps_command]
#     else:
#         return
#     subprocess.run(cmd) if blocking else subprocess.Popen(cmd)

# 初始化 LLM
client = OpenAI(
    api_key="sk-p4fUo6CD8qznKOLC8bQARL1BfcaulU7XCta9H0IJDNrAOXh3",  # 替换为你的 Moonshot Key
    base_url="https://api.moonshot.cn/v1",
)

def detect_intent(user_input: str):
    """调用大模型判断意图：任务指令 or 聊天"""
    prompt = f"""
你是一个机器人助手小智同学，请判断以下用户输入是属于“机器人任务规划指令”还是普通“聊天提问”：

输入：{user_input}

请返回一个JSON对象，格式如下：
{{
  "intent": "task" 或 "chat",
  "reason": "你做出判断的理由"
}}
"""
    response = client.chat.completions.create(
        model="moonshot-v1-8k",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
        max_tokens=200
    )
    content = response.choices[0].message.content.strip()
    try:
        json_start = content.find("{")
        return json.loads(content[json_start:])
    except:
        return {"intent": "chat", "reason": "意图识别失败，默认当作聊天"}

def load_predicted_skills(path="lerobot/resnet/data/llm/predicted_skills.json"):
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
        return data[0]["skill_sequence"]

def call_image_skill_predictor():
    try:
        subprocess.run(
            ["python", "lerobot/resnet/predict_image_skill.py", "--num-predicts", "1"],
            check=True
        )
    except Exception as e:
        st.error(f"图像技能识别失败: {e}")
        return []

    return load_predicted_skills()

def generate_prompt(task_description, predicted_skills):
    skill_dict = {
        0: "抓取熊猫（黑白）放到盒子",
        1: "抓取兔子（粉红色）放到盒子",
        2: "抓取老虎（黄色）放到盒子",
        3: "抓取蜘蛛侠（红色）放到盒子",
        4: "抓取公仔（蓝色）放到盒子",
        5: "无效/重复技能"
    }

    readable_skills = [skill_dict.get(s, f"未知技能{s}") for s in predicted_skills if s != 5]

    prompt = f"""
你是机器人任务规划助手小智同学，现在有如下任务：
- 你需要根据用户用自然语言描述的高层任务，结合当前环境状态，推理出最合理的技能执行序列。输出结构化的JSON格式的技能规划结果。
- 同时你具备对话能力，能用自然语言解释规划逻辑，回答用户的疑问（比如“你是谁，当前玩偶有什么”）。该类型判定为info
- 你理解技能库的能力边界，善于处理异常情况，确保规划合理且可执行。
- 输出的回答必须包含：JSON格式的任务规划结果，包含字段："status"（success/error/info）、"message"（简明提示）、"planned_skills"（技能列表，成功时非空，失败或信息时可空）。
-输出更加拟人，像真人一样回复
-如果检测到不止一个技能编号，根据用户的提问，匹配最合适的技能放到planned_skills，0: "抓取熊猫（黑白）放到盒子",
        1: "抓取兔子（粉红色）放到盒子",
        2: "抓取老虎（黄色）放到盒子",
        3: "抓取蜘蛛侠（红色）放到盒子",
        4: "抓取公仔（蓝色）放到盒子",
        5: "无效/重复技能"
- 用户描述任务为：{task_description}
- 当前图像识别出的可行技能为：{readable_skills}

请判断是否可以执行该任务，严格输出以下JSON格式：
{{
  "status": "success|error|info",
  "message": "对规划结果的解释",
  "planned_skills"（技能列表，格式为字符串数组，每项为“编号: 技能名称”，例如："0: 抓取熊猫（黑白）放到盒子"),
  "explanation": "这里是给用户的详细自然语言说明",
}}
请不要输出 JSON 以外的任何文本。
"""
    return prompt
import re
def extract_json(text):
    # 提取第一个完整的 JSON 对象
    match = re.search(r"\{[\s\S]*?\}", text)
    if match:
        return json.loads(match.group())
    else:
        raise ValueError("找不到有效的 JSON 内容")

def call_task_planner(task_description, predicted_skills):
    prompt = generate_prompt(task_description, predicted_skills)
    response = client.chat.completions.create(
        model="moonshot-v1-8k",
        messages=[
            {"role": "system", "content": "你是机器人任务规划助手小智同学。"},
            {"role": "user", "content": prompt}
        ],
        temperature=0.1,
        max_tokens=600,
    )
    answer = response.choices[0].message.content.strip()
    try:
        # json_start = answer.find("{")
        # return json.loads(answer[json_start:])
        return extract_json(answer)
    except Exception as e:
        return {
  "status": "error",
  "message": f"解析失败：{str(e)}",
  "raw_response": answer
}
        # return {"status": "error", "message": f"解析失败：{str(e)}", "raw_response": answer}

def call_chat_response(user_input):
    """非任务型对话时的回答"""
    response = client.chat.completions.create(
        model="moonshot-v1-8k",
        messages=[
            {"role": "system", "content": "你是机器人助手小智，具备亲和力和对话能力，能解释当前系统情况、图像状态、技能等"},
            {"role": "user", "content": user_input}
        ],
        temperature=0.3,
        max_tokens=500
    )
    return response.choices[0].message.content.strip()

# ---------- Streamlit 主界面 ----------
st.set_page_config(page_title="机器人任务规划器", layout="centered")
st.title("🤖 机器人智能助手小智")

task_description = st.text_input("请输入你的问题或任务：", "帮我整理熊猫")

if st.button("发送"):
    with st.spinner("🔍 正在分析输入意图..."):
        intent_result = detect_intent(task_description)
        st.info(f"判定为：{intent_result['intent']}  🎯（{intent_result['reason']}）")

    if intent_result["intent"] == "task":
        with st.spinner("🧠 正在根据环境匹配技能..."):

            predicted_skills = call_image_skill_predictor()

        if predicted_skills:
            with st.spinner("🤖 正在进行任务规划..."):
                result = call_task_planner(task_description, predicted_skills)
                st.subheader("📋 任务规划结果：")
                st.json(result)
                say(result.get("message", "任务已规划完成"), blocking=True)
                # 从 planned_skills 提取编号（支持 "0:xxx" 字符串或纯技能名）
                def extract_skill_ids_from_list(skills):
                    ids = []
                    for s in skills:
                        if isinstance(s, str):
                            match = re.match(r"(\d+)\s*:", s)
                            if match:
                                ids.append(int(match.group(1)))
                    return ids

                # 编号 → 指令映射
                skill_command_map = {
                    0: [
                        "python", "lerobot/scripts/gello_test.py", "record",
                        "--fps", "30", "--root", "data",
                        "--repo-id", "SunJincheng/gello_model",
                        "--tags", "tutorial", "eval",
                        "--warmup-time-s", "12", "--episode-time-s", "120",
                        "--reset-time-s", "30", "--num-episodes", "2",
                        "--force-override", "1",
                        "-p", "outputs/train/act_panda_dish/checkpoints/030000/pretrained_model"
                    ],
                    # 可添加更多技能编号及命令
                }

                def execute_skill_by_id(skill_id,planned_skills):
                    command = skill_command_map.get(skill_id)
                    if command:
                        try:
                            st.info(f"🛠️ 正在执行技能 {planned_skills}...")
                            subprocess.run(command, check=True)
                            st.success("✅ 技能执行完成。")
                        except subprocess.CalledProcessError as e:
                            st.error(f"❌ 技能执行失败：{e}")
                    else:
                        st.warning(f"⚠️ 未知技能编号：{skill_id}")

                # 执行阶段
                planned_skills = result.get("planned_skills", [])
                print("Planned skills:", planned_skills)
                skill_ids = extract_skill_ids_from_list(planned_skills)

                if skill_ids:
                    for skill_id in skill_ids:
                        execute_skill_by_id(skill_id,planned_skills)
                else:
                    st.warning("未提取到技能编号，跳过执行阶段。")
                
        else:
            st.error("未检测到可用技能，任务规划取消。")
    else:
        with st.spinner("💬 正在回复你的问题..."):
            response = call_chat_response(task_description)
            st.markdown(response)
            say(response)

with st.expander("📘 使用说明"):
    st.markdown("""
    输入可以是任务类指令（如“整理熊猫”），也可以是普通提问（如“你是谁”）。
    系统会自动识别意图，调用合适的处理方式：
    - **任务类** → 图像识别 + 技能规划
    - **聊天类** → 智能助手直接回复
    """)
