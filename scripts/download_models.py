import os
import shutil
from modelscope.hub.snapshot_download import snapshot_download

# ==============================================================================
# ==============================================================================
PROJECT_ROOT = "/root/autodl-tmp/GenRec_Explainer_Project"
MODELS_ROOT = os.path.join(PROJECT_ROOT, "models")
os.makedirs(MODELS_ROOT, exist_ok=True)

# ==============================================================================
# ==============================================================================
def smart_download(model_id, final_model_name, ignore_patterns=None):
    """
    Download a model while skipping unneeded file formats to save disk space.
    """
    final_save_path = os.path.join(MODELS_ROOT, final_model_name)
    
    if not os.path.exists(final_save_path):
        print(f"--- Processing model: {model_id} ---")
        
        temp_dir = os.path.join(PROJECT_ROOT, "temp_download_" + final_model_name)
        
        try:
            print(f"Downloading into temporary directory: {temp_dir}")
            print(f"Ignoring file patterns: {ignore_patterns}")
            
            snapshot_download(
                model_id=model_id,
                cache_dir=temp_dir,
                revision='master',
                ignore_file_pattern=ignore_patterns
            )
            print("Download succeeded.")

        except Exception as e:
            print(f"!!! Error during download: {e}")
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)
            return

        actual_model_path = temp_dir
        for root, dirs, files in os.walk(temp_dir):
            if "config.json" in files:
                actual_model_path = root
                break
        
        print(f"Detected actual model path: {actual_model_path}")
        print(f"Moving model to final path: {final_save_path}")
        
        if os.path.exists(actual_model_path):
            shutil.move(actual_model_path, final_save_path)
            print(f"Model prepared successfully at: {final_save_path}")
        else:
            print(f"!!! Error: downloaded model path not found.")

        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
            
    else:
        print(f"Model already exists at: {final_save_path}. Skipping download.")

# ==============================================================================
# ==============================================================================
if __name__ == "__main__":
    smart_download(
        model_id='google/flan-t5-xxl', 
        final_model_name='flan-t5-xxl',
        ignore_patterns=["*.msgpack", "*.safetensors", "*.h5"]
    )
    
    smart_download(
        model_id='facebook/bart-base', 
        final_model_name='bart-base',
        ignore_patterns=["*.msgpack", "*.safetensors", "*.h5"]
    )

    print("\nAll models are ready.")
