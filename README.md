# 语音处理服务

这是一个基于Faster Whisper和SpeechBrain的语音处理服务，提供以下功能：
- 语音转录（支持多语言，自动语言检测）
- 情绪识别
- 说话人识别
- 实时流式处理

## 环境要求

- Python 3.8+
- CUDA支持（可选，但推荐用于GPU加速）
- 足够的磁盘空间用于模型缓存

## 安装

1. 克隆仓库：
```bash
git clone [repository_url]
cd [repository_name]
```

2. 创建虚拟环境：
```bash
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# 或
.venv\Scripts\activate  # Windows
```

3. 安装依赖：
```bash
pip install -r requirements.txt
```

## 配置

1. 复制`.env.example`为`.env`：
```bash
cp .env.example .env
```

2. 根据需要修改`.env`中的配置：
- `WHISPER_MODEL_SIZE`: 选择模型大小（tiny/base/small/medium/large）
- `LANGUAGE`: 设置语言模式（"auto"为自动检测，也可指定"zh"或"en"等）

## 运行服务

```bash
python app.py
```

服务将在 http://localhost:8000 启动。

## API使用

### 1. 处理音频文件（非流式）

POST `/process_audio`

请求体：
- `audio_file`: 音频文件（支持wav格式）

响应示例：
```json
{
    "status": "success",
    "data": {
        "transcription": [
            {
                "text": "你好，hello world",
                "language": "zh",
                "start": 0.0,
                "end": 2.5
            },
            {
                "text": "This is a test",
                "language": "en",
                "start": 2.5,
                "end": 4.0
            }
        ],
        "emotion": "happy",
        "speaker_id": "speaker_1",
        "detected_language": "mixed"
    }
}
```

### 2. 流式处理（WebSocket）

WebSocket连接：`ws://localhost:8000/ws/stream`

客户端示例代码：
```javascript
const ws = new WebSocket('ws://localhost:8000/ws/stream');

ws.onopen = () => {
    console.log('连接已建立');
    // 发送音频数据
    ws.send(audioData);
};

ws.onmessage = (event) => {
    const data = JSON.parse(event.data);
    switch(data.type) {
        case 'transcription':
            console.log('实时转录:', data.text);
            console.log('检测到的语言:', data.detected_language);
            break;
        case 'emotion':
            console.log('情绪识别:', data.emotion);
            break;
        case 'speaker':
            console.log('说话人识别:', data.speaker_id);
            break;
        case 'final':
            console.log('最终结果:', data.data);
            break;
        case 'status':
            console.log('状态信息:', data.message);
            break;
        case 'error':
            console.error('错误:', data.message);
            break;
    }
};

ws.onerror = (error) => {
    console.error('WebSocket错误:', error);
};

ws.onclose = () => {
    console.log('连接已关闭');
};

// 发送停止信号
function stopProcessing() {
    ws.send('stop');
}
```

## 多语言支持说明

1. **自动语言检测**：
   - 默认使用自动语言检测模式（LANGUAGE="auto"）
   - 支持识别中英文混合的音频
   - 每个语音片段会标注对应的语言

2. **转录结果格式**：
   - `text`: 转录的文本内容
   - `language`: 该片段的语言（"zh"中文，"en"英文等）
   - `start`: 片段开始时间（秒）
   - `end`: 片段结束时间（秒）

3. **语言切换**：
   - 系统会自动检测语言的切换
   - 不同语言的片段会分开返回
   - 支持在同一段音频中混合使用多种语言

## 注意事项

1. 首次运行时会下载模型，这可能需要一些时间
2. 如果使用GPU，确保已正确安装CUDA和cuDNN
3. 对于中文场景，可能需要使用中文数据集重新训练情绪识别模型
4. 流式处理时，建议控制音频数据的发送频率，避免服务器过载
5. 发送停止信号后，服务器会立即停止处理并返回状态信息
6. 多语言混合音频的处理可能需要更大的模型（推荐使用base或更大的模型）

## 许可证

MIT 