from fastapi import APIRouter, Depends, HTTPException, UploadFile, File
from sqlalchemy.orm import Session
from typing import List, Optional
import numpy as np
from models.database import get_db, Speaker
from models.audio_processor import AudioProcessor
from models.cache import cache
from pydantic import BaseModel

router = APIRouter()
audio_processor = AudioProcessor()

class SpeakerBase(BaseModel):
    name: str
    description: Optional[str] = None
    gender: Optional[str] = None
    age: Optional[int] = None
    language: Optional[str] = None
    family_id: str

class SpeakerCreate(SpeakerBase):
    pass

class SpeakerResponse(SpeakerBase):
    id: int

    class Config:
        from_attributes = True

@router.post("/register", response_model=SpeakerResponse)
async def register_speaker(
    speaker_info: SpeakerCreate,
    audio_file: UploadFile = File(...),
    db: Session = Depends(get_db)
):
    """
    注册新的讲话人，需要提供个人信息和声音样本
    """
    # 检查名称是否已存在
    existing_speaker = db.query(Speaker).filter(Speaker.name == speaker_info.name).first()
    if existing_speaker:
        raise HTTPException(status_code=400, detail="讲话人名称已存在")
    
    # 保存上传的音频文件
    audio_path = f"temp/{audio_file.filename}"
    with open(audio_path, "wb") as buffer:
        content = await audio_file.read()
        buffer.write(content)
    
    try:
        # 提取声纹特征
        signal, fs = audio_processor.load_audio(audio_path)
        voice_embedding = audio_processor.extract_speaker_embedding(signal)
        
        # 创建讲话人记录
        db_speaker = Speaker(
            name=speaker_info.name,
            description=speaker_info.description,
            gender=speaker_info.gender,
            age=speaker_info.age,
            language=speaker_info.language,
            family_id=speaker_info.family_id,
            voice_embedding=voice_embedding.tolist()
        )
        
        db.add(db_speaker)
        db.commit()
        db.refresh(db_speaker)
        
        # 同步到Redis缓存
        speaker_data = {
            "id": db_speaker.id,
            "name": db_speaker.name,
            "description": db_speaker.description,
            "gender": db_speaker.gender,
            "age": db_speaker.age,
            "language": db_speaker.language,
            "family_id": db_speaker.family_id
        }
        cache.set_speaker(db_speaker.id, speaker_data, permanent=True)
        cache.set_embedding(db_speaker.id, voice_embedding, permanent=True)
        
        return db_speaker
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"处理音频时出错: {str(e)}")

@router.get("/list", response_model=List[SpeakerResponse])
def list_speakers(
    family_id: Optional[str] = None,
    db: Session = Depends(get_db)
):
    """
    获取讲话人列表，可以按家庭ID筛选
    """
    query = db.query(Speaker)
    if family_id:
        query = query.filter(Speaker.family_id == family_id)
    speakers = query.all()
    return speakers

@router.get("/{speaker_id}", response_model=SpeakerResponse)
def get_speaker(speaker_id: int, db: Session = Depends(get_db)):
    """
    获取特定讲话人的信息
    """
    speaker = db.query(Speaker).filter(Speaker.id == speaker_id).first()
    if not speaker:
        raise HTTPException(status_code=404, detail="讲话人不存在")
    return speaker

@router.delete("/{speaker_id}")
def delete_speaker(speaker_id: int, db: Session = Depends(get_db)):
    """
    删除讲话人信息
    """
    speaker = db.query(Speaker).filter(Speaker.id == speaker_id).first()
    if not speaker:
        raise HTTPException(status_code=404, detail="讲话人不存在")
    
    # 删除数据库记录
    db.delete(speaker)
    db.commit()
    
    # 删除Redis缓存
    cache.delete_speaker(speaker_id)
    
    return {"message": "讲话人已删除"}

@router.get("/family/{family_id}", response_model=List[SpeakerResponse])
def get_family_members(family_id: str, db: Session = Depends(get_db)):
    """
    获取指定家庭的所有成员
    """
    speakers = db.query(Speaker).filter(Speaker.family_id == family_id).all()
    return speakers

@router.post("/sync-cache")
def sync_cache(db: Session = Depends(get_db)):
    """
    同步数据库中的讲话人信息到Redis缓存
    """
    try:
        # 同步所有讲话人信息到Redis
        cache.sync_speakers_from_db(db)
        return {"message": "缓存同步成功"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"缓存同步失败: {str(e)}")

@router.post("/verify-cache")
def verify_cache(db: Session = Depends(get_db)):
    """
    验证Redis缓存中的讲话人信息是否在数据库中存在
    """
    try:
        # 验证并清理无效的缓存
        cache.verify_speakers(db)
        return {"message": "缓存验证完成"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"缓存验证失败: {str(e)}") 