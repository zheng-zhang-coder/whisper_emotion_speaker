import os

# 项目根目录
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# 模型缓存目录
MODEL_CACHE_DIR = os.path.join(BASE_DIR, 'model_cache')
EMOTION_MODEL_SAVE_DIR = os.path.join(MODEL_CACHE_DIR, 'emotion')

# 确保目录存在
os.makedirs(MODEL_CACHE_DIR, exist_ok=True)
os.makedirs(EMOTION_MODEL_SAVE_DIR, exist_ok=True)

# 音频处理配置
TARGET_SAMPLE_RATE = 16000  # wav2vec2 模型需要的采样率 