#!/bin/bash
echo "--- GenRec-E Training Script Started (From Scratch) ---"

# 定义我们环境的Python解释器的绝对路径
PYTHON_EXECUTABLE="/root/autodl-tmp/GenRec_Explainer_Project/envs/genrec_legacy/bin/python"

# 检查解释器是否存在
if [ ! -f "$PYTHON_EXECUTABLE" ]; then
    echo "!!! CRITICAL ERROR: Python executable not found. Aborting."
    exit 1
fi
echo "Using Python interpreter: ${PYTHON_EXECUTABLE}"

# ------------------- 核心修正！-------------------
# 我们把 GenRec 的主目录，也就是包含 genrec/ 这个子目录的文件夹，
# 强制地、手动地添加到Python的模块搜索路径中。
export PYTHONPATH="/root/autodl-tmp/GenRec_Explainer_Project/GenRec"
echo "Set PYTHONPATH to: ${PYTHONPATH}"
# --------------------------------------------------

# 我们现在可以从任何地方，用绝对路径来调用脚本，因为PYTHONPATH已经设定好了
# 我们为了清晰，还是先进入GenRec的主目录
cd /root/autodl-tmp/GenRec_Explainer_Project/GenRec
echo "Current directory: $(pwd)"

# 配置文件路径是相对于当前目录的
CONFIG_FILE_PATH="config/genrec_e_movielens_config.json"
echo "Starting training using config file: ${CONFIG_FILE_PATH}"
    
if [ ! -f "$CONFIG_FILE_PATH" ]; then
    echo "!!! CRITICAL ERROR: Config file not found. Aborting."
    exit 1
fi

# 我们调用的是 genrec/ 目录下的 train.py
${PYTHON_EXECUTABLE} genrec/train.py -c ${CONFIG_FILE_PATH}

# --- 检查结果的逻辑保持不变 ---
if [ $? -eq 0 ]; then
    echo "--- Training finished successfully. ---"
else
    echo "!!! ERROR: Training script exited with an error. ---"
fi