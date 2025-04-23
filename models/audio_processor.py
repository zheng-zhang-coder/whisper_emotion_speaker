import os
import numpy as np
import soundfile as sf
import torch
from config import settings
import asyncio
from sqlalchemy.orm import Session
from models.database import Speaker
from models.cache import cache
from models.model_manager import model_manager
from models.queue_manager import queue_manager
from typing import List, Tuple, Optional, Dict, Any
import logging

logger = logging.getLogger(__name__)

class AudioProcessor:
    def __init__(self):
        self.whisper_model = model_manager.get_model('whisper')
        self.speaker_model = model_manager.get_model('speaker')
        self.emotion_model = model_manager.get_model('emotion')
    
    def load_audio(self, audio_path: str) -> Tuple[np.ndarray, int]:
        """
        加载音频文件，支持PCM和WAV格式
        """
        if audio_path.endswith('.pcm'):
            # 读取PCM文件
            with open(audio_path, 'rb') as f:
                pcm_data = f.read()
            
            # 将PCM数据转换为numpy数组
            audio_data = np.frombuffer(pcm_data, dtype=np.int16)
            
            # 使用配置的采样率
            return audio_data, settings.SAMPLE_RATE
        else:
            # 读取WAV文件
            signal, fs = sf.read(audio_path)
            return signal, fs
    
    def save_pcm(self, audio_data: np.ndarray, output_path: str):
        """
        保存为PCM格式
        """
        # 确保数据是16位整数
        audio_data = (audio_data * 32767).astype(np.int16)
        with open(output_path, 'wb') as f:
            f.write(audio_data.tobytes())
    
    def extract_speaker_embedding(self, signal: np.ndarray) -> np.ndarray:
        """
        提取声纹特征向量
        """
        with torch.no_grad():
            # 确保信号是浮点数类型
            if signal.dtype != np.float32:
                signal = signal.astype(np.float32) / 32767.0
            
            embedding = self.speaker_model.encode_batch(torch.tensor(signal))
            return embedding.numpy()
    
    def compare_speakers(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """
        比较两个声纹特征向量的相似度
        """
        similarity = self.speaker_model.similarity(
            torch.tensor(embedding1),
            torch.tensor(embedding2)
        )
        return similarity.item()
    
    def identify_speaker(self, signal: np.ndarray, db: Session, family_id: Optional[str] = None) -> dict:
        """
        识别说话人，返回最匹配的说话人信息
        """
        # 提取当前音频的声纹特征
        current_embedding = self.extract_speaker_embedding(signal)
        
        # 获取所有已注册的说话人
        query = db.query(Speaker)
        if family_id:
            query = query.filter(Speaker.family_id == family_id)
        speakers = query.all()
        
        if not speakers:
            return {"speaker_id": None, "similarity": 0.0}
        
        # 计算与所有说话人的相似度
        max_similarity = 0.0
        best_match = None
        
        for speaker in speakers:
            # 尝试从缓存获取声纹特征
            cached_embedding = cache.get_embedding(speaker.id)
            if cached_embedding is None:
                # 如果缓存中没有，从数据库获取并缓存
                cached_embedding = np.array(speaker.voice_embedding)
                cache.set_embedding(speaker.id, cached_embedding)
            
            similarity = self.compare_speakers(
                current_embedding,
                cached_embedding
            )
            if similarity > max_similarity:
                max_similarity = similarity
                best_match = speaker
        
        # 如果相似度太低，认为是未知说话人
        if max_similarity < settings.SPEAKER_SIMILARITY_THRESHOLD:
            return {"speaker_id": None, "similarity": max_similarity}
        
        # 缓存说话人信息
        speaker_info = {
            "id": best_match.id,
            "name": best_match.name,
            "family_id": best_match.family_id,
            "similarity": max_similarity
        }
        cache.set_speaker(best_match.id, speaker_info)
        
        return speaker_info
    
    async def process_transcription(self, audio_data: bytes, metadata: Dict) -> Dict:
        """
        处理语音转录
        """
        try:
            # 保存临时文件
            temp_path = f"temp/{metadata.get('task_id', 'temp')}.pcm"
            with open(temp_path, 'wb') as f:
                f.write(audio_data)
            
            # 在线程池中运行语音转录
            transcription_future = model_manager.run_in_thread(
                self.whisper_model.transcribe,
                temp_path,
                language=settings.LANGUAGE,
                beam_size=settings.BEAM_SIZE,
                task="transcribe"
            )
            
            # 等待转录完成
            segments, info = await asyncio.wrap_future(transcription_future)
            
            # 处理转录结果
            transcription = []
            for segment in segments:
                transcription.append({
                    "text": segment.text,
                    "language": info.language,
                    "start": segment.start,
                    "end": segment.end
                })
            
            return {
                "type": "transcription",
                "data": transcription,
                "language": info.language
            }
            
        except Exception as e:
            logger.error(f"处理转录时出错: {str(e)}")
            raise
        finally:
            # 清理临时文件
            if os.path.exists(temp_path):
                os.remove(temp_path)
    
    async def process_emotion(self, audio_data: bytes, metadata: Dict) -> Dict:
        """
        处理情绪识别
        """
        try:
            # 保存临时文件
            temp_path = f"temp/{metadata.get('task_id', 'temp')}.pcm"
            with open(temp_path, 'wb') as f:
                f.write(audio_data)
            
            # 加载音频数据
            signal, fs = self.load_audio(temp_path)
            
            # 在线程池中运行情绪识别
            emotion_future = model_manager.run_in_thread(
                self.emotion_model.classify_batch,
                torch.tensor(signal)
            )
            emotion_scores = await asyncio.wrap_future(emotion_future)
            
            return {
                "type": "emotion",
                "data": emotion_scores[0].item()
            }
            
        except Exception as e:
            logger.error(f"处理情绪识别时出错: {str(e)}")
            raise
        finally:
            # 清理临时文件
            if os.path.exists(temp_path):
                os.remove(temp_path)
    
    async def process_speaker(self, audio_data: bytes, metadata: Dict) -> Dict:
        """
        处理说话人识别
        """
        try:
            # 保存临时文件
            temp_path = f"temp/{metadata.get('task_id', 'temp')}.pcm"
            with open(temp_path, 'wb') as f:
                f.write(audio_data)
            
            # 加载音频数据
            signal, fs = self.load_audio(temp_path)
            
            # 说话人识别
            speaker_info = {"speaker_id": None, "similarity": 0.0}
            if metadata.get('db'):
                speaker_info = self.identify_speaker(
                    signal,
                    metadata['db'],
                    metadata.get('family_id')
                )
            
            return {
                "type": "speaker",
                "data": speaker_info
            }
            
        except Exception as e:
            logger.error(f"处理说话人识别时出错: {str(e)}")
            raise
        finally:
            # 清理临时文件
            if os.path.exists(temp_path):
                os.remove(temp_path)
    
    async def process(self, audio_path: str, db: Session = None, family_id: Optional[str] = None):
        """
        处理音频文件，返回转录文本、情绪和说话人信息
        """
        try:
            # 读取音频数据
            with open(audio_path, 'rb') as f:
                audio_data = f.read()
            
            # 创建任务元数据
            metadata = {
                'db': db,
                'family_id': family_id
            }
            
            # 添加处理任务
            transcription_id = await queue_manager.add_audio_task(
                audio_data,
                'transcription',
                metadata
            )
            
            emotion_id = await queue_manager.add_audio_task(
                audio_data,
                'emotion',
                metadata
            )
            
            speaker_id = await queue_manager.add_audio_task(
                audio_data,
                'speaker',
                metadata
            )
            
            # 等待所有任务完成
            transcription_result = await queue_manager.get_task_result(transcription_id, 'transcription')
            emotion_result = await queue_manager.get_task_result(emotion_id, 'emotion')
            speaker_result = await queue_manager.get_task_result(speaker_id, 'speaker')
            
            return {
                "transcription": transcription_result['data'],
                "emotion": emotion_result['data'],
                "speaker_info": speaker_result['data'],
                "detected_language": transcription_result['language']
            }
            
        except Exception as e:
            logger.error(f"处理音频时出错: {str(e)}")
            raise
    
    async def process_stream(self, audio_path: str, db: Session = None, stream_id: str = None, family_id: Optional[str] = None):
        """
        流式处理音频文件，实时返回识别结果
        """
        try:
            # 读取音频数据
            with open(audio_path, 'rb') as f:
                audio_data = f.read()
            
            # 创建任务元数据
            metadata = {
                'db': db,
                'family_id': family_id,
                'stream_id': stream_id
            }
            
            # 添加处理任务
            transcription_id = await queue_manager.add_audio_task(
                audio_data,
                'transcription',
                metadata
            )
            
            # 等待转录结果
            while True:
                result = await queue_manager.get_task_result(transcription_id, 'transcription')
                if result:
                    yield result
                    break
                await asyncio.sleep(0.1)
            
            # 添加情绪识别任务
            emotion_id = await queue_manager.add_audio_task(
                audio_data,
                'emotion',
                metadata
            )
            
            # 等待情绪识别结果
            while True:
                result = await queue_manager.get_task_result(emotion_id, 'emotion')
                if result:
                    yield result
                    break
                await asyncio.sleep(0.1)
            
            # 添加说话人识别任务
            speaker_id = await queue_manager.add_audio_task(
                audio_data,
                'speaker',
                metadata
            )
            
            # 等待说话人识别结果
            while True:
                result = await queue_manager.get_task_result(speaker_id, 'speaker')
                if result:
                    yield result
                    break
                await asyncio.sleep(0.1)
            
        except Exception as e:
            logger.error(f"流式处理音频时出错: {str(e)}")
            raise 