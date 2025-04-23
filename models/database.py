from sqlalchemy import create_engine, Column, Integer, String, Float, ARRAY, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship
from config import settings

# 创建数据库引擎
engine = create_engine(settings.DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()

class Speaker(Base):
    __tablename__ = "speakers"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, unique=True, index=True)
    description = Column(String, nullable=True)
    # 存储声纹特征向量
    voice_embedding = Column(ARRAY(Float), nullable=False)
    # 可选的其他信息
    gender = Column(String, nullable=True)
    age = Column(Integer, nullable=True)
    language = Column(String, nullable=True)
    # 家庭分组ID
    family_id = Column(String, index=True, nullable=False)

# 创建数据库表
def init_db():
    Base.metadata.create_all(bind=engine)

# 获取数据库会话
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close() 