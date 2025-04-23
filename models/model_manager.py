import os
import torch
from faster_whisper import WhisperModel
from speechbrain.inference import SpeakerRecognition
from speechbrain.inference import Pretrained
from config import settings
import logging
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, Any
from models.custom_interface import CustomEncoderWav2vec2Classifier

logger = logging.getLogger(__name__)

class ModelManager:
    _instance = None
    _models: Dict[str, Any] = {}
    _executor = ThreadPoolExecutor(max_workers=settings.MAX_WORKERS)
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ModelManager, cls).__new__(cls)
        return cls._instance
    
    def __init__(self):
        if not self._models:
            self._initialize_models()
    
    def _initialize_models(self):
        """
        初始化所有模型
        """
        try:
            # 创建模型缓存目录
            os.makedirs(settings.MODEL_CACHE_DIR, exist_ok=True)
            
            # 初始化Whisper模型
            self._models['whisper'] = WhisperModel(
                settings.WHISPER_MODEL_SIZE,
                device=settings.DEVICE,
                compute_type=settings.COMPUTE_TYPE,
                download_root=settings.MODEL_CACHE_DIR
            )
            
            # 初始化SpeechBrain模型
            self._models['speaker'] = SpeakerRecognition.from_hparams(
                source=settings.SPEAKER_MODEL_PATH,
                savedir=os.path.join(settings.MODEL_CACHE_DIR, 'speaker')
            )
            
            # 初始化情感识别模型
            self._models['emotion'] = CustomEncoderWav2vec2Classifier.from_hparams(
                source="speechbrain/emotion-recognition-wav2vec2-IEMOCAP",
                savedir=settings.EMOTION_MODEL_SAVE_DIR
            )
            
            logger.info("所有模型初始化完成")
            
        except Exception as e:
            logger.error(f"模型初始化失败: {str(e)}")
            raise
    
    def get_model(self, model_name: str) -> Any:
        """
        获取指定的模型
        """
        return self._models.get(model_name)
    
    def run_in_thread(self, func, *args, **kwargs):
        """
        在线程池中运行函数
        """
        return self._executor.submit(func, *args, **kwargs)
    
    def shutdown(self):
        """
        关闭线程池
        """
        self._executor.shutdown(wait=True)

# 创建全局模型管理器实例
model_manager = ModelManager() 