import asyncio
from typing import Dict, Any, Optional
import logging
from config import settings
import uuid

logger = logging.getLogger(__name__)

class AudioQueueManager:
    _instance = None
    _queues: Dict[str, asyncio.Queue] = {}
    _processing_tasks: Dict[str, asyncio.Task] = {}
    _max_concurrent = settings.MAX_CONCURRENT_STREAMS
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(AudioQueueManager, cls).__new__(cls)
        return cls._instance
    
    def __init__(self):
        if not self._queues:
            self._initialize_queues()
    
    def _initialize_queues(self):
        """
        初始化处理队列
        """
        self._queues = {
            'transcription': asyncio.Queue(),
            'emotion': asyncio.Queue(),
            'speaker': asyncio.Queue()
        }
    
    async def add_audio_task(self, audio_data: bytes, task_type: str, metadata: Optional[Dict] = None) -> str:
        """
        添加音频处理任务
        :param audio_data: 音频数据
        :param task_type: 任务类型（transcription/emotion/speaker）
        :param metadata: 任务元数据
        :return: 任务ID
        """
        task_id = str(uuid.uuid4())
        task = {
            'id': task_id,
            'data': audio_data,
            'metadata': metadata or {}
        }
        
        await self._queues[task_type].put(task)
        logger.info(f"添加{task_type}任务: {task_id}")
        
        return task_id
    
    async def get_task_result(self, task_id: str, task_type: str) -> Optional[Dict]:
        """
        获取任务结果
        :param task_id: 任务ID
        :param task_type: 任务类型
        :return: 任务结果
        """
        # 检查任务是否在处理中
        if task_id in self._processing_tasks:
            try:
                result = await self._processing_tasks[task_id]
                return result
            except Exception as e:
                logger.error(f"获取任务结果失败: {str(e)}")
                return None
        return None
    
    async def process_queue(self, queue_name: str, processor_func):
        """
        处理队列中的任务
        :param queue_name: 队列名称
        :param processor_func: 处理函数
        """
        while True:
            try:
                # 获取任务
                task = await self._queues[queue_name].get()
                task_id = task['id']
                
                # 检查并发限制
                if len(self._processing_tasks) >= self._max_concurrent:
                    # 等待有任务完成
                    await asyncio.sleep(0.1)
                    continue
                
                # 创建处理任务
                self._processing_tasks[task_id] = asyncio.create_task(
                    processor_func(task['data'], task['metadata'])
                )
                
                # 等待任务完成
                try:
                    result = await self._processing_tasks[task_id]
                    logger.info(f"任务{task_id}处理完成")
                except Exception as e:
                    logger.error(f"任务{task_id}处理失败: {str(e)}")
                finally:
                    # 清理任务
                    del self._processing_tasks[task_id]
                    self._queues[queue_name].task_done()
                    
            except Exception as e:
                logger.error(f"处理队列{queue_name}时出错: {str(e)}")
                await asyncio.sleep(1)  # 出错时等待一段时间再继续
    
    def get_queue_status(self) -> Dict[str, int]:
        """
        获取队列状态
        :return: 各队列的任务数量
        """
        return {
            name: queue.qsize()
            for name, queue in self._queues.items()
        }
    
    def get_processing_status(self) -> Dict[str, int]:
        """
        获取处理状态
        :return: 正在处理的任务数量
        """
        return {
            'processing': len(self._processing_tasks),
            'max_concurrent': self._max_concurrent
        }

# 创建全局队列管理器实例
queue_manager = AudioQueueManager() 