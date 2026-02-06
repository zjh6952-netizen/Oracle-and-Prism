import os
import shutil
from modelscope.hub.snapshot_download import snapshot_download

# ==============================================================================
# 1. 配置部分 (CONFIGURATION)
# ==============================================================================
PROJECT_ROOT = "/root/autodl-tmp/GenRec_Explainer_Project"
MODELS_ROOT = os.path.join(PROJECT_ROOT, "models")
os.makedirs(MODELS_ROOT, exist_ok=True)

# ==============================================================================
# 2. 智能下载函数
# ==============================================================================
def smart_download(model_id, final_model_name, ignore_patterns=None):
    """
    一个智能的下载函数，可以忽略不需要的文件格式，从源头节省空间。
    """
    final_save_path = os.path.join(MODELS_ROOT, final_model_name)
    
    if not os.path.exists(final_save_path):
        print(f"--- 开始处理模型: {model_id} ---")
        
        # 为了绝对干净，我们还是用一个临时目录
        temp_dir = os.path.join(PROJECT_ROOT, "temp_download_" + final_model_name)
        
        try:
            # 【核心优化】使用 ignore_file_pattern 参数
            # 我们告诉 ModelScope：在下载时，请跳过所有.msgpack, .safetensors, .h5 文件
            print(f"将要下载到临时目录: {temp_dir}")
            print(f"将忽略以下文件格式: {ignore_patterns}")
            
            snapshot_download(
                model_id=model_id,
                cache_dir=temp_dir,
                revision='master',
                ignore_file_pattern=ignore_patterns
            )
            print("下载成功。")

        except Exception as e:
            print(f"!!! 下载过程中发生错误: {e}")
            # 清理失败的临时文件夹
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)
            return

        # 找到真正包含 config.json 的文件夹
        actual_model_path = temp_dir
        for root, dirs, files in os.walk(temp_dir):
            if "config.json" in files:
                actual_model_path = root
                break
        
        print(f"找到实际模型文件路径: {actual_model_path}")
        print(f"正在移动到最终路径: {final_save_path}")
        
        # 将正确的文件夹移动到最终位置
        if os.path.exists(actual_model_path):
            shutil.move(actual_model_path, final_save_path)
            print(f"模型已成功准备并存放到: {final_save_path}")
        else:
            print(f"!!! 错误：找不到下载好的模型文件路径！")

        # 清理可能残留的空的上级临时目录
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
            
    else:
        print(f"模型已存在于: {final_save_path}，跳过下载。")

# ==============================================================================
# 3. 主程序 (MAIN LOGIC)
# ==============================================================================
if __name__ == "__main__":
    # --- 下载教师模型，只保留.bin文件 ---
    smart_download(
        model_id='google/flan-t5-xxl', 
        final_model_name='flan-t5-xxl',
        ignore_patterns=["*.msgpack", "*.safetensors", "*.h5"] # 忽略其他所有框架的文件
    )
    
    # --- 下载学生模型，也只保留.bin文件 ---
    smart_download(
        model_id='facebook/bart-base', 
        final_model_name='bart-base',
        ignore_patterns=["*.msgpack", "*.safetensors", "*.h5"] # 同样应用忽略规则
    )

    print("\n🎉 所有模型准备完毕！")