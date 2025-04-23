from pydantic_settings import BaseSettings
import os
import torch
import logging

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

class Settings(BaseSettings):
    # 模型设置
    WHISPER_MODEL_SIZE: str = "medium"  # 可选: tiny, base, small, medium, large
    DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"
    COMPUTE_TYPE: str = "float16" if torch.cuda.is_available() else "float32"
    
    # 语言设置
    LANGUAGE: str = "auto"  # 自动检测语言
    SUPPORTED_LANGUAGES: list = ["zh", "en"]  # 支持的语言列表
    
    # SpeechBrain模型路径
    SPEAKER_MODEL_PATH: str = "speechbrain/spkrec-ecapa-voxceleb"
    EMOTION_MODEL_PATH: str = "speechbrain/emotion-recognition-wav2vec2-IEMOCAP"
    
    # 模型缓存和临时文件目录
    MODEL_CACHE_DIR: str = "model_cache"
    EMOTION_MODEL_SAVE_DIR: str = os.path.join(MODEL_CACHE_DIR, "emotion")
    TEMP_DIR: str = "temp"  # 临时文件目录
    TEMP_FILE_TTL: int = 3600  # 临时文件保留时间（秒）
    
    # 音频设置
    TARGET_SAMPLE_RATE: int = 16000  # wav2vec2 模型要求的采样率
    SAMPLE_RATE: int = 16000  # 采样率
    CHANNELS: int = 1  # 声道数
    BITS_PER_SAMPLE: int = 16  # 位深度
    MAX_AUDIO_LENGTH: int = 600  # 最大音频长度（秒）
    MIN_AUDIO_LENGTH: int = 1  # 最小音频长度（秒）
    AUDIO_CHUNK_SIZE: int = 30  # 音频分块大小（秒）
    
    # 推理设置
    BEAM_SIZE: int = 5
    
    # 数据库设置
    DATABASE_URL: str = "postgresql://zixiai_dev:dI8lY3wQ2gZ9lF6fG5@postgres-e58a77d9fa06-custom-115c-public.rds-pg.volces.com:5432/zixiai_dev"
    
    # Redis设置
    REDIS_HOST: str = "redis-shzl6ittlnwxl3ama.redis.volces.com"
    REDIS_PORT: int = 6379
    REDIS_DB: int = 0
    REDIS_PASSWORD: str = "%406%40pAbQdK4P2sjk"
    REDIS_USER: str = "zixiai_dev"  # 修改为空字符串而不是 None

    # 缓存设置
    CACHE_TTL: int = 3600  # 缓存过期时间（秒）
    SPEAKER_CACHE_PREFIX: str = "speaker:"  # 说话人缓存前缀
    EMBEDDING_CACHE_PREFIX: str = "embedding:"  # 声纹特征缓存前缀
    
    # 说话人识别设置
    SPEAKER_SIMILARITY_THRESHOLD: float = 0.75  # 说话人识别的相似度阈值
    
    # 并发设置
    MAX_WORKERS: int = 4  # 线程池大小
    MAX_CONCURRENT_STREAMS: int = 10  # 最大并发流数
    MAX_CONCURRENT_REQUESTS: int = 20  # 最大并发请求数
    
    # 服务器设置
    HOST: str = "0.0.0.0"
    PORT: int = 8000  # 添加服务器端口设置
    
    # HuggingFace设置
    HF_TOKEN: str = ""
    
    class Config:
        env_file = ".env"
        extra = "allow"  # 允许额外的字段

settings = Settings()

# 创建必要的目录
os.makedirs(settings.MODEL_CACHE_DIR, exist_ok=True)
os.makedirs("temp", exist_ok=True) 