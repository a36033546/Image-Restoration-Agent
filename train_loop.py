import os
import random
import cv2
import base64
import shutil
import numpy as np
from skimage.metrics import peak_signal_noise_ratio as psnr
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage

# ========== 設定路徑 ==========

PROJECT_DIR = r"C:\Users\jimmy\OneDrive\Desktop\RL_Agent_Project"

DATASET = {
    "rainy": {
        "input": os.path.join(PROJECT_DIR, "dataset", "rainy", "input"),
        "gt":    os.path.join(PROJECT_DIR, "dataset", "rainy", "gt"),
    },
    "blurry": {
        "input": os.path.join(PROJECT_DIR, "dataset", "blurry", "input"),
        "gt":    os.path.join(PROJECT_DIR, "dataset", "blurry", "gt"),
    },
    "hazy": {
        "input": os.path.join(PROJECT_DIR, "dataset", "hazy", "input"),
        "gt":    os.path.join(PROJECT_DIR, "dataset", "hazy", "gt"),
    },
}

TEMP_INPUT    = os.path.join(PROJECT_DIR, "temp_input",  "current.png")
TEMP_ENHANCED = os.path.join(PROJECT_DIR, "temp_input",  "enhanced.png")
TEMP_OUTPUT   = os.path.join(PROJECT_DIR, "temp_output", "current.png")

# ========== RL Policy State（Multi-armed Bandit）==========
# 每種退化類型有兩個選項：
#   選項 0 = 直接修復（不做前處理）
#   選項 1 = 前處理後再修復
# PSNR 高 → 這個選項權重上升 → 下次更常被選
# PSNR 低 → 這個選項權重下降 → 下次較少被選

policy_state = {
    "rainy":  {"weights": [1.0, 1.0], "rewards": [[], []], "correct": 0, "total": 0},
    "blurry": {"weights": [1.0, 1.0], "rewards": [[], []], "correct": 0, "total": 0},
    "hazy":   {"weights": [1.0, 1.0], "rewards": [[], []], "correct": 0, "total": 0},
}

LEARNING_RATE = 0.1
REWARD_BASE   = 20.0

# ========== 前處理函式 ==========

def preprocess_rainy(image_path: str, output_path: str):
    """去雨前處理：CLAHE 對比增強，讓雨滴更明顯以利模型辨識"""
    img = cv2.imread(image_path)
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    enhanced = cv2.merge([l, a, b])
    enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
    cv2.imwrite(output_path, enhanced)


def preprocess_blurry(image_path: str, output_path: str):
    """去模糊前處理：Unsharp Mask 銳化，強化邊緣讓模型更好恢復細節"""
    img = cv2.imread(image_path)
    blurred = cv2.GaussianBlur(img, (0, 0), 3)
    sharpened = cv2.addWeighted(img, 1.5, blurred, -0.5, 0)
    cv2.imwrite(output_path, sharpened)


def preprocess_hazy(image_path: str, output_path: str):
    """去霧前處理：直方圖均衡化，提升對比度讓霧氣特徵更清楚"""
    img = cv2.imread(image_path)
    ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    y, cr, cb = cv2.split(ycrcb)
    y = cv2.equalizeHist(y)
    enhanced = cv2.merge([y, cr, cb])
    enhanced = cv2.cvtColor(enhanced, cv2.COLOR_YCrCb2BGR)
    cv2.imwrite(output_path, enhanced)


# ========== 工具函式 ==========

def get_random_image():
    degradation = random.choice(["rainy", "blurry", "hazy"])
    input_dir = DATASET[degradation]["input"]
    gt_dir    = DATASET[degradation]["gt"]
    files = [f for f in os.listdir(input_dir) if f.lower().endswith(('.png', '.jpg'))]
    fname = random.choice(files)
    return degradation, os.path.join(input_dir, fname), os.path.join(gt_dir, fname)


def calculate_reward(gt_path: str, output_path: str) -> float:
    try:
        img_gt     = cv2.imread(gt_path)
        img_output = cv2.imread(output_path)
        if img_gt is None or img_output is None:
            return -10.0
        if img_gt.shape != img_output.shape:
            img_output = cv2.resize(img_output, (img_gt.shape[1], img_gt.shape[0]))
        return psnr(img_gt, img_output)
    except Exception as e:
        print(f"❌ 評分失敗：{e}")
        return -10.0


def copy_image(src, dst):
    import shutil
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    shutil.copy2(src, dst)


def image_to_base64(image_path):
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def select_action(degradation: str) -> int:
    """Multi-armed Bandit：根據權重選擇動作 0（直接修復）或 1（前處理後修復）"""
    weights = policy_state[degradation]["weights"]
    total   = sum(weights)
    probs   = [w / total for w in weights]
    r = random.random()
    return 0 if r <= probs[0] else 1


def update_policy(degradation: str, action: int, reward: float, correct: bool):
    """用 PSNR reward 更新對應動作的權重"""
    state = policy_state[degradation]
    state["rewards"][action].append(reward)
    state["total"] += 1
    if correct:
        state["correct"] += 1

    # 正規化 reward：高於基準線為正向，低於為負向
    normalized = (reward - REWARD_BASE) / REWARD_BASE

    # PSNR 低於 20 額外加重懲罰，讓系統更快學會避開不好的選擇
    if reward < REWARD_BASE:
        normalized *= 2.0

    state["weights"][action] += LEARNING_RATE * normalized
    state["weights"][action]  = max(0.1, state["weights"][action])

    action_name = "直接修復" if action == 0 else "前處理+修復"
    print(f"  🎯 動作：{action_name} | reward：{reward:.2f} | 權重更新 → {state['weights'][action]:.3f}")
    total_w = sum(state["weights"])
    probs   = [f'{w/total_w*100:.0f}%' for w in state["weights"]]
    print(f"  📊 {degradation} 選用機率：直接修復={probs[0]}  前處理+修復={probs[1]}")

# ========== 工具箱 ==========

from test_agent import apply_derain_model, apply_deblur_model, apply_dehaze_model

TOOLS = {
    "rainy":  apply_derain_model,
    "blurry": apply_deblur_model,
    "hazy":   apply_dehaze_model,
}

PREPROCESS = {
    "rainy":  preprocess_rainy,
    "blurry": preprocess_blurry,
    "hazy":   preprocess_hazy,
}

# ========== 單一 Episode ==========

def run_episode(episode_num: int):
    # 1. 隨機取一張圖
    degradation, input_path, gt_path = get_random_image()

    # 2. llava 判斷退化類型
    copy_image(input_path, TEMP_INPUT)
    img_b64 = image_to_base64(TEMP_INPUT)

    vision_prompt = """Look at this image carefully.
Describe what kind of degradation you see:
- Do you see rain streaks or raindrops?
- Is the image blurry or out of focus?
- Is there haze, fog, or low visibility?
Answer in one sentence."""

    message = HumanMessage(content=[
        {"type": "image_url", "image_url": f"data:image/png;base64,{img_b64}"},
        {"type": "text", "text": vision_prompt}
    ])

    response     = llava_llm.invoke([message])
    answer_lower = response.content.strip().lower()

    # 3. 計分制解析退化類型
    rain_score   = sum(1 for w in ["rain", "streak", "drop", "wet"] if w in answer_lower)
    hazy_score   = sum(1 for w in ["haz", "fog", "mist", "smoke", "visibility", "white"] if w in answer_lower)
    blurry_score = sum(1 for w in ["blur", "motion", "focus", "sharp"] if w in answer_lower)
    scores       = {"rainy": rain_score, "hazy": hazy_score, "blurry": blurry_score}

    if max(scores.values()) > 0:
        chosen_degradation = max(scores, key=scores.get)
    else:
        chosen_degradation = "blurry"  # 預設

    correct = (chosen_degradation == degradation)

    # 4. RL Policy 選擇動作（0=直接修復，1=前處理後修復）
    action = select_action(degradation)
    action_name = "直接修復" if action == 0 else "前處理+修復"

    weights    = policy_state[degradation]["weights"]
    total_w    = sum(weights)
    probs      = [f'{w/total_w*100:.0f}%' for w in weights]
    print(f"\n[Episode {episode_num}] 退化：{degradation} | 圖片：{os.path.basename(input_path)}")
    print(f"    llava 判斷：{answer_lower[:60]} → {chosen_degradation} {'✅' if correct else '❌'}")
    print(f"   RL 選擇動作：{action_name} | 機率：直接={probs[0]} 前處理={probs[1]}")

    # 5. 執行
    if action == 1:
        PREPROCESS[chosen_degradation](TEMP_INPUT, TEMP_ENHANCED)
        execute_input = TEMP_ENHANCED
    else:
        execute_input = TEMP_INPUT

    tool   = TOOLS[chosen_degradation]
    result = tool.invoke({"image_path": execute_input, "output_path": TEMP_OUTPUT})
    print(f"  🔧 {result}")

    save_dir = os.path.join(PROJECT_DIR, "results", f"ep{episode_num:03d}_{degradation}")
    os.makedirs(save_dir, exist_ok=True)
    shutil.copy2(TEMP_OUTPUT, os.path.join(save_dir, "output.png"))
    shutil.copy2(TEMP_INPUT,  os.path.join(save_dir, "input.png"))
    shutil.copy2(gt_path,     os.path.join(save_dir, "gt.png"))
    print(f"  💾 結果已儲存至 results/ep{episode_num:03d}_{degradation}/")

    # 6. 計算 reward
    reward = calculate_reward(gt_path, TEMP_OUTPUT)
    print(f"  🏆 PSNR reward：{reward:.2f}")

    # 7. 只有 LLaVA 判斷正確才更新 RL 權重
    if correct:
        update_policy(degradation, action, reward, correct)
        print("   LLaVA 判斷正確，RL 已更新權重。")
    else:
        print("   LLaVA 判斷錯誤，跳過 RL 更新。")

    return reward, chosen_degradation, action_name, correct


# ========== 主訓練迴圈 ==========

def train(num_episodes: int = 60):
    history = []

    for ep in range(1, num_episodes + 1):
        reward, degradation, action_name, correct = run_episode(ep)
        history.append({
            "episode":     ep,
            "degradation": degradation,
            "action":      action_name,
            "reward":      reward,
            "correct":     correct,
        })

    print("\n========== 訓練結果 ==========")
    rewards = [h["reward"] for h in history]
    print(f"平均 PSNR reward：{sum(rewards)/len(rewards):.2f}")
    print(f"最高 PSNR reward：{max(rewards):.2f}")
    print(f"最低 PSNR reward：{min(rewards):.2f}")

    correct_count = sum(1 for h in history if h["correct"])
    print(f"退化類型判斷正確率：{correct_count}/{num_episodes} ({correct_count/num_episodes*100:.0f}%)")

    print("\n========== Policy 最終狀態 ==========")
    for deg, state in policy_state.items():
        avg_r0 = sum(state["rewards"][0]) / len(state["rewards"][0]) if state["rewards"][0] else 0
        avg_r1 = sum(state["rewards"][1]) / len(state["rewards"][1]) if state["rewards"][1] else 0
        total_w = sum(state["weights"])
        probs   = [f'{w/total_w*100:.0f}%' for w in state["weights"]]
        print(f"{deg:6s} | 直接修復={probs[0]} (avg PSNR {avg_r0:.2f}) | 前處理+修復={probs[1]} (avg PSNR {avg_r1:.2f})")

    print("\n========== 詳細記錄 ==========")
    for h in history:
        c = "✅" if h["correct"] else "❌"
        print(f"Episode {h['episode']:02d} | {h['degradation']:6s} | {h['action']:10s} | PSNR {h['reward']:.2f} | {c}")

    return history


# ========== 使用模式 ==========

def restore_image(input_path: str, output_path: str):
    """使用者丟一張圖進來，系統自動判斷並修復"""
    print(f"\n🔍 分析圖片：{input_path}")

    img_b64 = image_to_base64(input_path)

    vision_prompt = """Look at this image carefully.
Describe what kind of degradation you see:
- Do you see rain streaks or raindrops?
- Is the image blurry or out of focus?
- Is there haze, fog, or low visibility?
Answer in one sentence."""

    message = HumanMessage(content=[
        {"type": "image_url", "image_url": f"data:image/png;base64,{img_b64}"},
        {"type": "text", "text": vision_prompt}
    ])

    response     = llava_llm.invoke([message])
    answer_lower = response.content.strip().lower()
    print(f"👁️  llava 判斷：{answer_lower}")

    rain_score   = sum(1 for w in ["rain", "streak", "drop", "wet"] if w in answer_lower)
    hazy_score   = sum(1 for w in ["haz", "fog", "mist", "smoke", "visibility", "white"] if w in answer_lower)
    blurry_score = sum(1 for w in ["blur", "motion", "focus", "sharp"] if w in answer_lower)
    scores       = {"rainy": rain_score, "hazy": hazy_score, "blurry": blurry_score}
    print(f" 得分：{scores}")

    chosen_key = max(scores, key=scores.get) if max(scores.values()) > 0 else "blurry"

    # 使用模式：選用訓練後權重較高的動作
    action = select_action(chosen_key)
    action_name = "直接修復" if action == 0 else "前處理+修復"
    print(f" 退化類型：{chosen_key} | 動作：{action_name}")

    copy_image(input_path, TEMP_INPUT)
    if action == 1:
        PREPROCESS[chosen_key](TEMP_INPUT, TEMP_ENHANCED)
        execute_input = TEMP_ENHANCED
    else:
        execute_input = TEMP_INPUT

    result = TOOLS[chosen_key].invoke({"image_path": execute_input, "output_path": output_path})
    print(f" {result}")
    print(f" 修復完成，輸出至：{output_path}")


# ========== 進入點 ==========

if __name__ == "__main__":
    import sys
    llava_llm = ChatOllama(model="llava", temperature=0)

    if len(sys.argv) > 1 and sys.argv[1] == "restore":
        if len(sys.argv) < 4:
            print("用法：python train_loop.py restore <輸入路徑> <輸出路徑>")
        else:
            restore_image(sys.argv[2], sys.argv[3])
    else:
        train(num_episodes=60)