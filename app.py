from fastapi import FastAPI, UploadFile, File, WebSocket, Depends, HTTPException, Query, Request, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
import uvicorn
from models.audio_processor import AudioProcessor
from models.database import init_db, get_db
from models.cache import cache
from models.model_manager import model_manager
from models.queue_manager import queue_manager
from routes import speaker
from config import settings
import json
import os
from sqlalchemy.orm import Session
from typing import Optional
import logging
import asyncio
import wave
import torchaudio
import torch
import numpy as np
from pathlib import Path
from fastapi import WebSocketDisconnect
from fastapi.responses import RedirectResponse
from contextlib import asynccontextmanager

logger = logging.getLogger(__name__)

# 创建必要的目录
os.makedirs("temp", exist_ok=True)
os.makedirs("speaker_samples", exist_ok=True)
os.makedirs("static", exist_ok=True)
os.makedirs("templates", exist_ok=True)
os.makedirs(settings.MODEL_CACHE_DIR, exist_ok=True)
os.makedirs(settings.EMOTION_MODEL_SAVE_DIR, exist_ok=True)

@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        # 初始化数据库
        init_db()
        
        # 初始化并发限制
        cache.set_concurrent_limit("stream", 0)
        
        # 初始化模型（会自动下载）
        logger.info("正在初始化模型...")
        model_manager._initialize_models()
        logger.info("模型初始化完成")
        
        # 启动队列处理
        asyncio.create_task(queue_manager.process_queue('transcription', audio_processor.process_transcription))
        asyncio.create_task(queue_manager.process_queue('emotion', audio_processor.process_emotion))
        asyncio.create_task(queue_manager.process_queue('speaker', audio_processor.process_speaker))
        logger.info("队列处理已启动")
        
        yield
    except Exception as e:
        logger.error(f"启动时出错: {str(e)}")
        raise
    finally:
        try:
            # 关闭模型管理器
            model_manager.shutdown()
            logger.info("模型管理器已关闭")
        except Exception as e:
            logger.error(f"关闭时出错: {str(e)}")

app = FastAPI(title="语音处理服务", lifespan=lifespan)

# 配置CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 配置模板
templates = Jinja2Templates(directory="templates")

# 配置静态文件
app.mount("/static", StaticFiles(directory="static"), name="static")

# 初始化音频处理器
audio_processor = AudioProcessor()

# 包含说话人管理路由
app.include_router(speaker.router, prefix="/speakers", tags=["speakers"])

@app.get("/")
async def home(request: Request):
    """
    返回注册页面
    """
    return templates.TemplateResponse("register.html", {"request": request})

@app.post("/process_audio")
async def process_audio(
    audio_file: UploadFile = File(...),
    family_id: Optional[str] = Query(None, description="家庭ID"),
    db: Session = Depends(get_db)
):
    """
    处理上传的音频文件，返回转录文本、情绪和说话人信息
    """
    try:
        # 保存上传的音频文件
        audio_path = f"temp/{audio_file.filename}"
        with open(audio_path, "wb") as buffer:
            content = await audio_file.read()
            buffer.write(content)
        
        # 处理音频
        result = await audio_processor.process(audio_path, db, family_id)
        
        return {
            "status": "success",
            "data": result
        }
    except Exception as e:
        logger.error(f"处理音频时出错: {str(e)}")
        return {
            "status": "error",
            "message": str(e)
        }

@app.websocket("/ws/stream")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    logger.info("WebSocket连接已建立")
    
    try:
        audio_buffer = []
        audio_params = None
        
        while True:
            try:
                # 接收消息
                message = await websocket.receive()
                
                # 如果是二进制数据
                if "bytes" in message:
                    if not audio_params:
                        raise ValueError("未收到音频参数")
                    
                    # 将二进制数据添加到缓冲区
                    audio_buffer.append(message["bytes"])
                    
                # 如果是文本数据（JSON）
                elif "text" in message:
                    data = json.loads(message["text"])
                    
                    if data["type"] == "audio_params":
                        audio_params = {
                            "channels": data["channels"],
                            "sample_width": data["sample_width"],
                            "frame_rate": data["frame_rate"],
                            "speaker_group": data.get("speaker_group", None)
                        }
                        logger.info(f"接收到音频参数: {audio_params}")
                        
                    elif data["type"] == "end_of_stream":
                        if not audio_buffer:
                            raise ValueError("未收到音频数据")
                        
                        # 处理完整的音频数据
                        audio_data = b"".join(audio_buffer)
                        
                        # 保存为临时WAV文件
                        temp_wav = "temp/temp_stream.wav"
                        with wave.open(temp_wav, "wb") as wf:
                            wf.setnchannels(audio_params["channels"])
                            wf.setsampwidth(audio_params["sample_width"])
                            wf.setframerate(audio_params["frame_rate"])
                            wf.writeframes(audio_data)
                        
                        # 加载音频文件
                        waveform, sample_rate = torchaudio.load(temp_wav)
                        
                        # 如果是立体声，转换为单声道
                        if waveform.shape[0] > 1:
                            waveform = torch.mean(waveform, dim=0, keepdim=True)
                        
                        # 重采样到16kHz（如果需要）
                        if sample_rate != 16000:
                            resampler = torchaudio.transforms.Resample(sample_rate, 16000)
                            waveform = resampler(waveform)
                        
                        try:
                            # 语音识别
                            whisper_model = model_manager.get_model("whisper")
                            segments, _ = whisper_model.transcribe(temp_wav)
                            transcription = " ".join([segment.text for segment in segments])
                            
                            # 发送转写结果
                            await websocket.send_json({
                                "type": "transcription",
                                "text": transcription
                            })
                            
                            # 情感分析
                            emotion_model = model_manager.get_model("emotion")
                            emotion_out, emotion_score, _, _ = emotion_model.classify_file(temp_wav)
                            
                            # 发送情感分析结果
                            await websocket.send_json({
                                "type": "emotion",
                                "emotion": emotion_out,
                                "confidence": emotion_score
                            })
                            
                            # 只有当提供了说话人分组信息时才进行说话人识别
                            if audio_params.get("speaker_group"):
                                # 说话人识别
                                speaker_model = model_manager.get_model("speaker")
                                current_embedding = speaker_model.encode_batch(waveform)
                                
                                # 寻找最匹配的说话人
                                best_score = -1
                                best_speaker = "未知说话人"
                                
                                # 遍历所有已注册的说话人
                                speaker_dir = Path("speaker_samples")
                                for embedding_file in speaker_dir.glob(f"{audio_params['speaker_group']}/*.npy"):
                                    speaker_name = embedding_file.stem
                                    speaker_embedding = np.load(embedding_file)
                                    
                                    # 计算余弦相似度
                                    similarity = torch.nn.functional.cosine_similarity(
                                        current_embedding.cpu(),
                                        torch.from_numpy(speaker_embedding),
                                        dim=-1
                                    ).item()
                                    
                                    if similarity > best_score and similarity > settings.SPEAKER_SIMILARITY_THRESHOLD:
                                        best_score = similarity
                                        best_speaker = speaker_name
                                
                                # 发送说话人识别结果
                                await websocket.send_json({
                                    "type": "speaker",
                                    "speaker_id": best_speaker,
                                    "confidence": best_score
                                })
                            
                            # 发送处理完成信号
                            await websocket.send_json({
                                "type": "end_of_processing"
                            })
                            
                        except Exception as e:
                            logger.error(f"处理音频时出错: {str(e)}")
                            await websocket.send_json({
                                "type": "error",
                                "message": str(e)
                            })
                        
                        # 清空缓冲区
                        audio_buffer = []
                        audio_params = None
                    
            except WebSocketDisconnect:
                logger.info("WebSocket连接已关闭")
                break
            except Exception as e:
                logger.error(f"处理WebSocket消息时出错: {str(e)}")
                try:
                    await websocket.send_json({
                        "type": "error",
                        "message": str(e)
                    })
                except:
                    break
                
    except Exception as e:
        logger.error(f"WebSocket处理时出错: {str(e)}")
    finally:
        try:
            await websocket.close()
        except:
            pass

@app.get("/queue/status")
async def get_queue_status():
    """
    获取队列状态
    """
    return {
        "queue_status": queue_manager.get_queue_status(),
        "processing_status": queue_manager.get_processing_status()
    }

@app.get("/reload")
async def reload():
    """强制重新加载页面的路由"""
    return RedirectResponse(url="/", headers={
        "Cache-Control": "no-cache, no-store, must-revalidate",
        "Pragma": "no-cache",
        "Expires": "0"
    })

@app.post("/register_speaker")
async def register_speaker(name: str = Form(...), audio: UploadFile = File(...)):
    try:
        # 保存音频文件
        audio_path = f"speaker_samples/{name}.wav"
        with open(audio_path, "wb") as f:
            content = await audio.read()
            f.write(content)
        
        # 加载音频文件
        waveform, sample_rate = torchaudio.load(audio_path)
        
        # 如果是立体声，转换为单声道
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        
        # 重采样到16kHz（如果需要）
        if sample_rate != 16000:
            resampler = torchaudio.transforms.Resample(sample_rate, 16000)
            waveform = resampler(waveform)
        
        # 提取说话人特征
        speaker_model = model_manager.get_model('speaker')
        embedding = speaker_model.encode_batch(waveform)
        
        # 保存说话人特征
        np.save(f"speaker_samples/{name}.npy", embedding.cpu().numpy())
        
        return {"success": True, "message": "说话人注册成功"}
    except Exception as e:
        logger.error(f"注册说话人失败: {str(e)}")
        return {"success": False, "message": str(e)}

@app.post("/recognize")
async def recognize_audio(audio: UploadFile = File(...)):
    try:
        # 保存音频文件
        audio_path = "temp/input.wav"
        with open(audio_path, "wb") as f:
            content = await audio.read()
            f.write(content)
        
        # 加载音频文件
        waveform, sample_rate = torchaudio.load(audio_path)
        
        # 如果是立体声，转换为单声道
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        
        # 重采样到16kHz（如果需要）
        if sample_rate != 16000:
            resampler = torchaudio.transforms.Resample(sample_rate, 16000)
            waveform = resampler(waveform)
        
        # 获取模型
        whisper_model = model_manager.get_model('whisper')
        emotion_model = model_manager.get_model('emotion')
        speaker_model = model_manager.get_model('speaker')
        
        # 转写音频
        segments, _ = whisper_model.transcribe(audio_path)
        transcription = " ".join([segment.text for segment in segments])
        
        # 情感分析
        emotion_out, emotion_score = emotion_model.classify_file(audio_path)
        
        # 说话人识别
        current_embedding = speaker_model.encode_batch(waveform)
        
        # 寻找最匹配的说话人
        best_score = -1
        best_speaker = "未知说话人"
        
        # 遍历所有已注册的说话人
        speaker_dir = Path("speaker_samples")
        for embedding_file in speaker_dir.glob("*.npy"):
            speaker_name = embedding_file.stem
            speaker_embedding = np.load(embedding_file)
            
            # 计算余弦相似度
            similarity = torch.nn.functional.cosine_similarity(
                current_embedding.cpu(),
                torch.from_numpy(speaker_embedding),
                dim=-1
            ).item()
            
            if similarity > best_score and similarity > settings.SPEAKER_SIMILARITY_THRESHOLD:
                best_score = similarity
                best_speaker = speaker_name
        
        return {
            "transcription": transcription,
            "emotion": emotion_out,
            "emotion_confidence": emotion_score,
            "speaker": best_speaker,
            "speaker_confidence": best_score
        }
    except Exception as e:
        logger.error(f"识别失败: {str(e)}")
        return {"error": str(e)}

if __name__ == "__main__":
    uvicorn.run("app:app", host="127.0.0.1", port=8080, reload=True) 