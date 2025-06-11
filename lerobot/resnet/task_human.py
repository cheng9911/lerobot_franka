# streamlit run task_human.py --server.address=192.168.1.
import streamlit as st
import json
# import pyttsx3
from openai import OpenAI
import queue
import threading
from gtts import gTTS
from playsound import playsound

# streamlit run task_human.py
# 初始化语音引擎
# tts_engine = pyttsx3.init()
# tts_engine.setProperty('rate', 150)
# tts_engine.setProperty('volume', 1.0)
#
#
# speech_queue = queue.Queue()
import os
import platform
import subprocess


def say(text, blocking=False):
    system = platform.system()

    if system == "Darwin":
        cmd = ['say', text]
        if not blocking:
            # macOS 用 & 实现后台执行，这里用 subprocess.Popen 实现非阻塞
            subprocess.Popen(cmd)
            return
        else:
            subprocess.run(cmd)

    elif system == "Linux":
        cmd = ['espeak-ng', '-v', 'cmn-latn-pinyin', text]
        if not blocking:
            subprocess.Popen(cmd)
            return
        else:
            subprocess.run(cmd)

    elif system == "Windows":
        # PowerShell 语音合成命令，注意双引号和单引号的转义
        # 用 start /b 实现后台执行
        ps_command = (
            f'Add-Type -AssemblyName System.Speech;'
            f'(New-Object System.Speech.Synthesis.SpeechSynthesizer).Speak("{text}")'
        )
        if not blocking:
            # start /b 启动后台进程
            cmd = ['powershell', '-WindowStyle', 'Hidden', '-Command', f'start /b powershell -Command "{ps_command}"']
            subprocess.Popen(cmd, shell=True)
        else:
            cmd = ['powershell', '-Command', ps_command]
            subprocess.run(cmd, shell=True)

    else:
        raise NotImplementedError(f"Unsupported platform: {system}")



# 初始化 OpenAI 客户端
client = OpenAI(
    api_key="sk-p4fUo6CD8qznKOLC8bQARL1BfcaulU7XCta9H0IJDNrAOXh3",  # 替换为你的API KEY
    base_url="https://api.moonshot.cn/v1",
)


# def generate_prompt(task_description, environment_json):
#     return f"""
# 你是一个高级机器人任务规划助手，精通环境感知与任务分解。
# 你需要根据用户用自然语言描述的高层任务，结合当前环境状态，推理出最合理的技能执行序列。
# 你理解技能库的能力边界，善于处理异常情况，确保规划合理且可执行。
# 你的回答必须严格符合指定的JSON格式，方便程序解析。请用简洁明了的语言反馈规划结果，避免多余信息。
#
# ## 技能库（可用技能）：
# - 抓取熊猫放到盒子
# - 抓取兔子放到盒子
# - 抓取老虎放到盒子
# - 抓取蜘蛛侠放到盒子
# - 抓取公仔放到盒子
#
# ## 输入格式：
#
# 任务描述（自然语言）：{task_description}
#
# 视觉环境状态（JSON格式）：
# {environment_json}
#
# ## 输出格式（JSON）示例：
#
# 成功示例：
# {{
#   "status": "success",
#   "message": "共发现N个未整理的玩偶，已生成整理计划",
#   "planned_skills": [
#     "抓取老虎放到盒子",
#     "抓取兔子放到盒子"
#   ]
# }}
#
# 失败示例（目标不存在）：
# {{
#   "status": "error",
#   "message": "目标玩偶熊猫不存在于当前环境，无法执行整理任务"
# }}
#
# 失败示例（技能缺失）：
# {{
#   "status": "error",
#   "message": "技能库中不存在处理目标 '可乐' 的技能，请添加对应技能"
# }}
#
# 请根据上述规则，结合任务描述和环境状态，严格输出对应的JSON规划结果。
# """
def generate_prompt(task_description, environment_json):
    return f"""
你是一个机器人高级任务规划助手小智同学，职责是：
- 你需要根据用户用自然语言描述的高层任务，结合当前环境状态，推理出最合理的技能执行序列。输出结构化的JSON格式的技能规划结果。
- 同时你具备对话能力，能用自然语言解释规划逻辑，回答用户的疑问（比如“你是谁，当前玩偶有什么”）。该类型判定为info
- 你理解技能库的能力边界，善于处理异常情况，确保规划合理且可执行。
- 输出的回答必须包含：JSON格式的任务规划结果，包含字段："status"（success/error/info）、"message"（简明提示）、"planned_skills"（技能列表，成功时非空，失败或信息时可空）。
-输出更加拟人，像真人一样回复


技能库（可用技能）：
- 0 抓取熊猫（黑白）放到盒子
- 1 抓取兔子（粉红色）放到盒子
- 2 抓取老虎（黄色）放到盒子
- 3 抓取蜘蛛侠（红色）放到盒子
- 4 抓取公仔（蓝色）放到盒子

输入：
任务描述（自然语言）：{task_description}

环境状态（JSON格式）：
{environment_json}

请严格按照以下格式输出：

JSON:
{{
  "status": "success|error|info",
  "message": "详细的状态消息",
  "planned_skills": [技能字符串列表]
  “skills_description”: "技能描述，成功时非空，失败或信息时可空"
}}


注意：
- 如果任务中提及的目标不存在于环境，status设为"error"，message说明具体原因，planned_skills为空。
- 如果任务中提及技能库没有的技能，status设为"error"，message提示添加技能，planned_skills为空。
- 如果任务是普通对话或询问（如“你是谁”），status设为"info"，message给出身份描述，planned_skills为空。

请根据上述规则，结合任务描述和环境状态，严格输出对应的JSON规划结果。
"""



def call_task_planner(task_description, environment_json):
    prompt = generate_prompt(task_description, environment_json)

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
        json_start = answer.find("{")
        result_json = json.loads(answer[json_start:])
        return result_json
    except Exception as e:
        return {"status": "error", "message": f"解析失败：{str(e)}", "raw_response": answer}


# Streamlit 界面
st.set_page_config(page_title="机器人任务规划器", layout="centered")
st.title("🤖 机器人智能任务规划器")

# 用户输入
task_description = st.text_input("请输入你的任务描述：", "帮我整理熊猫")
environment_json_text = st.text_area("请输入环境状态（JSON格式）",
                                     '''{
                                       "objects": [
                                         {"name": "老虎", "exists": true, "in_box": false},
                                         {"name": "兔子", "exists": true, "in_box": false},
                                         {"name": "蜘蛛侠", "exists": true, "in_box": false}
                                       ]
                                     }''', height=250)

# 显示按钮和响应
if st.button("提交任务"):
    try:
        env_data = json.loads(environment_json_text)
        result = call_task_planner(task_description, json.dumps(env_data, ensure_ascii=False))
        st.subheader("📋 任务规划结果：")
        st.json(result)
        # speak(result.get("message", "任务执行完成。"))
        say(result.get("message", "任务执行完成。"), blocking=True)
    except Exception as e:
        st.error(f"输入错误：{e}")
        # speak("环境状态解析失败，请检查JSON格式。")
        say("环境状态解析失败，请检查JSON格式。", blocking=True)

# 展示提示示例和帮助信息
with st.expander("🧠 提示示例和说明"):
    st.markdown("""
    **示例任务描述：**
    - 帮我整理全部玩偶
    - 整理蜘蛛侠
    - 给我熊猫

    **示例环境状态：**
    ```json
    {
      "objects": [
        {"name": "熊猫", "exists": false, "in_box": false},
        {"name": "兔子", "exists": true, "in_box": false},
        {"name": "老虎", "exists": true, "in_box": true}
      ]
    }
    ```

    **注意：** JSON 格式必须严格，字段 `name`, `exists`, `in_box` 都必须包含。
    """)
