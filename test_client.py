import asyncio
import websockets
import json
import wave
import argparse
import logging
from pathlib import Path
import csv

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

async def stream_audio(websocket, audio_file, speaker_group=None):
    """
    流式发送音频数据
    """
    try:
        with wave.open(str(audio_file), 'rb') as wf:
            # 获取音频参数
            channels = wf.getnchannels()
            sample_width = wf.getsampwidth()
            frame_rate = wf.getframerate()

            # 发送音频参数
            params = {
                'type': 'audio_params',
                'channels': channels,
                'sample_width': sample_width,
                'frame_rate': frame_rate
            }
            if speaker_group:
                params['speaker_group'] = speaker_group

            await websocket.send(json.dumps(params))

            # 读取并发送音频数据
            chunk_size = 81920  # 每次读取1KB
            while True:
                data = wf.readframes(chunk_size)
                if not data:
                    break

                # 直接发送二进制数据
                await websocket.send(data)

            # 发送结束标记
            await websocket.send(json.dumps({
                'type': 'end_of_stream'
            }))

    except Exception as e:
        logger.error(f"发送音频数据时出错: {str(e)}")
        raise

async def receive_results(websocket):
    """
    接收并打印识别结果
    """
    results = []
    try:
        while True:
            response = await websocket.recv()
            result = json.loads(response)

            if result['type'] == 'transcription':
                results.append({'type': 'transcription', 'text': result['text']})
            elif result['type'] == 'emotion':
                results.append({'type': 'emotion', 'emotion': result['emotion']})
            elif result['type'] == 'speaker':
                results.append({'type': 'speaker', 'speaker_id': result['speaker_id']})
            elif result['type'] == 'error':
                results.append({'type': 'error', 'message': result['message']})
            elif result['type'] == 'end_of_processing':
                break

    except Exception as e:
        logger.error(f"接收结果时出错: {str(e)}")
        raise

    return results

async def process_audio_file(websocket, audio_file, speaker_group=None):
    send_task = asyncio.create_task(stream_audio(websocket, audio_file, speaker_group))
    receive_task = asyncio.create_task(receive_results(websocket))

    await asyncio.gather(send_task, receive_task)
    return await receive_task

def save_results_to_csv(results, file_name, output_file):
    with open(output_file, mode='a', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        for result in results:
            if result['type'] == 'transcription':
                writer.writerow([file_name, result['text']])
            elif result['type'] == 'emotion':
                writer.writerow([file_name, f"情感分析: {result['emotion']}"])
            elif result['type'] == 'speaker':
                writer.writerow([file_name, f"说话人识别: {result['speaker_id']}"])
            elif result['type'] == 'error':
                writer.writerow([file_name, f"错误: {result['message']}"])

async def main():
    parser = argparse.ArgumentParser(description='测试流式音频识别客户端')
    parser.add_argument('--host', default='localhost', help='服务器主机地址')
    parser.add_argument('--port', type=int, default=8000, help='服务器端口')
    parser.add_argument('--folder', required=True, help='音频文件夹路径')
    parser.add_argument('--output-csv', required=True, help='输出CSV文件路径')
    parser.add_argument('--speaker-group', help='说话人分组')
    args = parser.parse_args()

    # 检查音频文件夹是否存在
    audio_folder = Path(args.folder)
    if not audio_folder.exists() or not audio_folder.is_dir():
        logger.error(f"音频文件夹不存在或不是一个目录: {args.folder}")
        return

    # 连接WebSocket服务器
    uri = f"ws://{args.host}:{args.port}/ws/stream"
    try:
        async with websockets.connect(uri) as websocket:
            logger.info(f"已连接到服务器: {uri}")

            # 递归遍历文件夹中的所有wav文件
            for audio_file in audio_folder.rglob('*.wav'):
                logger.info(f"处理文件: {audio_file}")
                results = await process_audio_file(websocket, audio_file, args.speaker_group)
                save_results_to_csv(results, audio_file.name, args.output_csv)

    except Exception as e:
        logger.error(f"连接服务器时出错: {str(e)}")

if __name__ == "__main__":
    asyncio.run(main())
