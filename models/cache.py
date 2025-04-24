import redis
import json
import numpy as np
from config import settings
from typing import Optional, Any, List
from sqlalchemy.orm import Session
from models.database import Speaker
import logging

logger = logging.getLogger(__name__)

class RedisCache:
    def __init__(self):
        try:
            self.redis = redis.Redis(
                host=settings.REDIS_HOST,
                port=settings.REDIS_PORT,
                db=settings.REDIS_DB,
                password=settings.REDIS_PASSWORD,
                username=settings.REDIS_USER,
                decode_responses=True,
                socket_timeout=5,
                socket_connect_timeout=5
            )
            # 测试连接
            self.redis.ping()
            logger.info("Redis连接成功")
        except Exception as e:
            logger.error(f"Redis连接失败: {str(e)}")
            raise
    
    def set_speaker(self, speaker_id: int, speaker_data: dict, permanent: bool = False):
        """
        缓存说话人信息
        :param speaker_id: 说话人ID
        :param speaker_data: 说话人数据
        :param permanent: 是否永久缓存
        """
        key = f"asr-speaker-{speaker_id}"
        if permanent:
            self.redis.set(key, json.dumps(speaker_data))
        else:
            self.redis.setex(
                key,
                settings.CACHE_TTL,
                json.dumps(speaker_data)
            )
    
    def get_speaker(self, speaker_id: int) -> Optional[dict]:
        """
        获取缓存的说话人信息
        """
        key = f"asr-speaker-{speaker_id}"
        data = self.redis.get(key)
        return json.loads(data) if data else None
    
    def set_embedding(self, speaker_id: int, embedding: np.ndarray, permanent: bool = False):
        """
        缓存声纹特征向量
        :param speaker_id: 说话人ID
        :param embedding: 声纹特征向量
        :param permanent: 是否永久缓存
        """
        key = f"asr-embedding-{speaker_id}"
        if permanent:
            self.redis.set(key, json.dumps(embedding.tolist()))
        else:
            self.redis.setex(
                key,
                settings.CACHE_TTL,
                json.dumps(embedding.tolist())
            )
    
    def get_embedding(self, speaker_id: int) -> Optional[np.ndarray]:
        """
        获取缓存的声纹特征向量
        """
        key = f"asr-embedding-{speaker_id}"
        data = self.redis.get(key)
        return np.array(json.loads(data)) if data else None
    
    def delete_speaker(self, speaker_id: int):
        """
        删除说话人缓存
        """
        speaker_key = f"asr-speaker-{speaker_id}"
        embedding_key = f"asr-embedding-{speaker_id}"
        self.redis.delete(speaker_key, embedding_key)
    
    def sync_speakers_from_db(self, db: Session):
        """
        从数据库同步说话人信息到Redis
        """
        speakers = db.query(Speaker).all()
        for speaker in speakers:
            # 缓存说话人信息
            speaker_data = {
                "id": speaker.id,
                "name": speaker.name,
                "description": speaker.description,
                "gender": speaker.gender,
                "age": speaker.age,
                "language": speaker.language
            }
            self.set_speaker(speaker.id, speaker_data, permanent=True)
            
            # 缓存声纹特征
            self.set_embedding(speaker.id, np.array(speaker.voice_embedding), permanent=True)
    
    def verify_speakers(self, db: Session):
        """
        验证Redis中的说话人信息是否在数据库中存在
        """
        # 获取所有缓存的说话人ID
        speaker_keys = self.redis.keys("asr-speaker-*")
        for key in speaker_keys:
            speaker_id = int(key.split("-")[-1])
            # 检查数据库是否存在该说话人
            speaker = db.query(Speaker).filter(Speaker.id == speaker_id).first()
            if not speaker:
                # 如果数据库不存在，删除缓存
                self.delete_speaker(speaker_id)
    
    def set_stream_result(self, stream_id: str, result: dict):
        """
        缓存流式处理结果
        """
        key = f"stream:{stream_id}"
        self.redis.setex(
            key,
            settings.CACHE_TTL,
            json.dumps(result)
        )
    
    def get_stream_result(self, stream_id: str) -> Optional[dict]:
        """
        获取缓存的流式处理结果
        """
        key = f"stream:{stream_id}"
        data = self.redis.get(key)
        return json.loads(data) if data else None
    
    def delete_stream_result(self, stream_id: str):
        """
        删除流式处理结果
        """
        key = f"stream:{stream_id}"
        self.redis.delete(key)
    
    def set_concurrent_limit(self, key: str, limit: int):
        """
        设置并发限制
        """
        self.redis.setex(
            f"limit:{key}",
            settings.CACHE_TTL,
            str(limit)
        )
    
    def check_concurrent_limit(self, key: str) -> bool:
        """
        检查并发限制
        """
        current = int(self.redis.get(f"limit:{key}") or 0)
        if current >= settings.MAX_CONCURRENT_STREAMS:
            return False
        self.redis.incr(f"limit:{key}")
        return True
    
    def release_concurrent_limit(self, key: str):
        """
        释放并发限制
        """
        self.redis.decr(f"limit:{key}")

# 创建全局缓存实例
cache = RedisCache() 