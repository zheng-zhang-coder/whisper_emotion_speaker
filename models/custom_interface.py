import torch
import torchaudio
from speechbrain.inference import Pretrained
import logging
import json

logger = logging.getLogger(__name__)

class CustomEncoderWav2vec2Classifier(Pretrained):
    # 情感标签映射
    EMOTION_MAP = {
        # 基本情绪
        "ang": "愤怒",
        "hap": "开心",
        "neu": "中性",
        "sad": "悲伤",
        "exc": "兴奋",
        "fru": "沮丧",
        "fea": "恐惧",
        "sur": "惊讶",
        "dis": "厌恶",
        "oth": "其他",
        
        # IEMOCAP 数据集情绪
        "angry": "愤怒",
        "happy": "开心",
        "neutral": "中性",
        "sad": "悲伤",
        "excited": "兴奋",
        "frustrated": "沮丧",
        "fearful": "恐惧",
        "surprised": "惊讶",
        "disgusted": "厌恶",
        "other": "其他",
        
        # RAVDESS 数据集情绪
        "calm": "平静",
        "happy": "开心",
        "sad": "悲伤",
        "angry": "愤怒",
        "fearful": "恐惧",
        "disgust": "厌恶",
        "surprised": "惊讶",
        "neutral": "中性",
        
        # CREMA-D 数据集情绪
        "anger": "愤怒",
        "disgust": "厌恶",
        "fear": "恐惧",
        "happy": "开心",
        "neutral": "中性",
        "sad": "悲伤",
        
        # SAVEE 数据集情绪
        "anger": "愤怒",
        "disgust": "厌恶",
        "fear": "恐惧",
        "happiness": "开心",
        "neutral": "中性",
        "sadness": "悲伤",
        "surprise": "惊讶",
        
        # 复合情绪
        "anxious": "焦虑",
        "bored": "无聊",
        "confused": "困惑",
        "contempt": "轻蔑",
        "embarrassed": "尴尬",
        "guilty": "愧疚",
        "jealous": "嫉妒",
        "proud": "自豪",
        "relaxed": "放松",
        "shame": "羞愧",
        "tired": "疲惫",
        "worried": "担忧"
    }
    
    def __init__(self, modules, hparams, **kwargs):
        super().__init__(modules, hparams, **kwargs)
        self.sample_rate = 16000  # 设置采样率
        
        # 打印所有可用的组件
        logger.info("Available modules:")
        for key in modules.keys():
            logger.info(f"- {key}")
            
        # 打印 hparams 的内容
        logger.info("Hparams content:")
        logger.info(json.dumps(hparams, indent=2, default=str))
        
        # 加载模型组件
        self.wav2vec2 = modules["wav2vec2"]
        self.avg_pool = modules["avg_pool"]
        self.classifier = modules["output_mlp"]  # 使用 output_mlp 作为分类器
        
        # 获取标签编码器
        if "label_encoder" in hparams:
            self.label_encoder = hparams["label_encoder"]
            # 设置标签编码器的期望长度
            self.label_encoder.expect_len(4)  # 使用 4 种基本情绪类别
        else:
            # 如果没有标签编码器，创建一个简单的映射
            self.label_encoder = {
                "decode_ndim": lambda x: ["ang", "hap", "neu", "sad"][x]
            }
            logger.warning("Using default emotion labels: angry, happy, neutral, sad")
        
    def classify_file(self, path):
        """
        对音频文件进行情感分类
        Args:
            path: 音频文件路径
        Returns:
            tuple: (预测的情感标签, 置信度分数, 预测索引, 文本标签)
        """
        waveform, sr = torchaudio.load(path)
        if sr != self.sample_rate:
            waveform = torchaudio.transforms.Resample(sr, self.sample_rate)(waveform)
            
        # 确保音频是单声道
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
            
        with torch.no_grad():
            wav_lens = torch.tensor([1.0])
            pred = self.classify_batch(waveform, wav_lens)
            
        # 获取预测结果
        prob = torch.softmax(pred[0], dim=-1)
        score, index = torch.max(prob, dim=-1)
        emotion_en = self.label_encoder.decode_ndim(index.item())
        emotion_cn = self.EMOTION_MAP.get(emotion_en, "未知")
        
        return emotion_cn, score.item(), index.item(), emotion_cn
        
    def classify_batch(self, waveform, wav_lens):
        """
        对音频波形进行情感分类
        Args:
            waveform: 音频波形张量
            wav_lens: 音频长度张量
        Returns:
            tensor: 预测结果
        """
        with torch.no_grad():
            # 确保波形在正确的设备上
            waveform = waveform.to(self.device)
            wav_lens = wav_lens.to(self.device)
            
            # 使用模型进行预测
            outputs = self.wav2vec2(waveform)
            embeddings = self.avg_pool(outputs, wav_lens)
            predictions = self.classifier(embeddings)
            
            return predictions 