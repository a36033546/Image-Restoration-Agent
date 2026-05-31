# 影像修復 Agent：結合 RL 提示優化與 Python 多工具協調

> 結合 RL 提示優化與 Python 多工具協調  
> 方向 A（自我進化）+ 方向 B（多工具調用）  

---

## 專題簡介

本系統設計一個影像修復 Agent，能自動辨識影像退化類型（去雨、去模糊、去霧），並透過 Multi-armed Bandit 強化學習自動學習最佳修復策略。

系統核心流程：

```
輸入退化影像
    ↓
LLaVA 視覺判斷（判斷退化類型）
    ↓
RL Policy 選擇策略（直接修復 or 前處理後修復）
    ↓
深度學習修復模型執行（Restormer / AOD-Net）
    ↓
PSNR 計算獎勵 → 更新 Policy 權重（防火牆機制）
    ↓
輸出修復影像
```

---

## 環境需求

### 作業系統
- Windows 10 / 11（本專題在 Windows 11 開發與測試）

### Python 版本
```
Python 3.12.6
```

### 必要套件
```
torch>=2.0.0
torchvision
opencv-python>=4.8.0
langchain-ollama>=0.1.0
langchain-core>=0.1.0
scikit-image>=0.21.0
numpy>=1.24.0
PyYAML>=6.0
```

### 本地 LLM（LLaVA）
需安裝 [Ollama](https://ollama.com) 並下載 LLaVA 模型：
```bash
# 安裝 Ollama 後執行
ollama pull llava
```

---

## 安裝步驟

### 1. Clone 專案

```bash
git clone https://github.com/（你的帳號）/RL_Agent_Project.git
cd RL_Agent_Project
```

### 2. 安裝 Python 套件

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install opencv-python langchain-ollama langchain-core scikit-image numpy PyYAML
```

### 3. 下載預訓練模型權重

#### Restormer（去雨 + 去模糊）
從 [Restormer GitHub](https://github.com/swz30/Restormer) 下載，放置於以下路徑：

```
RL_Agent_Project/
├── Deraining/
│   ├── Options/Deraining_Restormer.yml
│   └── pretrained_models/deraining.pth
└── Motion_Deblurring/
    ├── Options/Deblurring_Restormer.yml
    └── pretrained_models/motion_deblurring.pth
```

#### AOD-Net（去霧）
下載 `dehazer.pth` 並放置於：

```
RL_Agent_Project/
└── Dehazing/
    └── pretrained_models/dehazer.pth
```

### 4. 建立資料集資料夾結構

```
RL_Agent_Project/
└── dataset/
    ├── rainy/
    │   ├── input/     ← 放退化影像（有雨的圖）
    │   └── gt/        ← 放對應清晰原圖
    ├── blurry/
    │   ├── input/     ← 放退化影像（模糊的圖）
    │   └── gt/        ← 放對應清晰原圖
    └── hazy/
        ├── input/     ← 放退化影像（有霧的圖）
        └── gt/        ← 放對應清晰原圖
```

建議資料集來源：
- **去雨**：[Rain100L](https://www.icst.pku.edu.cn/struct/Projects/joint_rain_removal.html)
- **去模糊**：用程式批次生成（見下方說明）
- **去霧**：[RESIDE SOTS Outdoor](https://sites.google.com/view/reside-dehaze-datasets/reside-v0)

#### 批次生成模糊圖片

```python
import cv2, os, numpy as np

gt_dir    = r"dataset\blurry\gt"      # 放清晰圖片
input_dir = r"dataset\blurry\input"   # 自動產生模糊圖片
os.makedirs(input_dir, exist_ok=True)

for fname in os.listdir(gt_dir):
    img = cv2.imread(os.path.join(gt_dir, fname))
    size = 30
    kernel = np.zeros((size, size))
    kernel[int((size-1)/2), :] = np.ones(size) / size
    blurred = cv2.filter2D(img, -1, kernel)
    cv2.imwrite(os.path.join(input_dir, fname), blurred)
```

### 5. 建立暫存資料夾

```bash
mkdir temp_input
mkdir temp_output
```

### 6. 安裝 Ollama 並啟動 LLaVA

```bash
# 下載並安裝 Ollama：https://ollama.com/download
ollama pull llava
ollama serve   # 背景啟動（通常安裝後會自動執行）
```

---

## 執行方式

### 訓練模式（執行 60 個 episodes 的 RL 訓練）

```bash
python train_loop.py
```

訓練過程會顯示每個 episode 的：
- LLaVA 判斷結果
- RL 策略選擇
- PSNR 獎勵
- Policy 權重更新

訓練完成後，所有修復結果會儲存至：

```
RL_Agent_Project/results/
├── ep001_rainy/
│   ├── input.png     ← 退化原圖
│   ├── output.png    ← 修復後圖片
│   └── gt.png        ← 清晰原圖（對比用）
├── ep002_hazy/
...
```

### 使用模式（單張影像自動修復）

```bash
python train_loop.py restore <輸入圖片路徑> <輸出圖片路徑>
```

範例：

```bash
python train_loop.py restore C:\Users\...\rainy_photo.png C:\Users\...\result.png
```

系統會自動：
1. 判斷影像退化類型
2. 根據訓練後的 Policy 選擇最佳策略
3. 執行修復並輸出結果

---

## 專案檔案說明

```
RL_Agent_Project/
├── test_agent.py       # 三個修復工具函式（去雨、去模糊、去霧）
├── train_loop.py       # 主訓練迴圈 + RL 機制 + 使用模式
├── dataset/            # 訓練資料集
├── results/            # 訓練過程中儲存的修復結果
├── temp_input/         # 暫存輸入影像
├── temp_output/        # 暫存輸出影像
├── Deraining/          # Restormer 去雨模型
├── Motion_Deblurring/  # Restormer 去模糊模型
├── Dehazing/           # AOD-Net 去霧模型
└── basicsr/            # Restormer 依賴的自訂套件
```

---

## 系統架構

| 模組 | 功能 | 技術 |
|---|---|---|
| 視覺感知 | 判斷影像退化類型 | LLaVA + 計分制關鍵字解析 |
| 前處理 | 可選的影像前處理 | OpenCV（CLAHE / Unsharp Mask / Histogram EQ）|
| 修復工具 | 深度學習影像修復 | Restormer / AOD-Net（PyTorch）|
| RL Policy | 選擇最佳修復策略 | Multi-armed Bandit（比例加權選擇）|
| 獎勵計算 | 評估修復品質 | PSNR（scikit-image）|
| 防火牆 | 保護 RL 學習信號 | 感知正確性過濾機制 |

---

## 實驗結果

| 退化類型 | 直接修復 PSNR | 前處理 PSNR | RL 最終偏好 |
|---|---|---|---|
| rainy  | 29.64 dB | 14.19 dB | 直接修復 72% |
| blurry | 18.91 dB | 19.35 dB | 前處理 49% |
| hazy   | 19.66 dB | 15.49 dB | 直接修復 59% |

整體退化類型判斷正確率：**65%**（60 episodes）

---

## 注意事項

- Restormer 推論使用CPU 模式，每次修復約需 1-3 分鐘
- LLaVA 需要約 8GB RAM 才能順暢運行
- 所有推論完全在本地端執行，不需要網路連線
