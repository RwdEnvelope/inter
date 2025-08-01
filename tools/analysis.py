# tools/av_tools.py

from langchain.tools import tool
from tools.audio_analysis import RecorderController
from tools.video_analysis import VideoController
import threading

class AVController:
    def __init__(self):
        self.audio = RecorderController()
        self.video = VideoController()
        self.threads = []

    def start(self):
        t_audio = threading.Thread(target=self.audio.start)
        t_video = threading.Thread(target=self.video.start)
        t_audio.start()
        t_video.start()
        self.threads = [t_audio, t_video]

    def stop(self):
        audio_path = self.audio.stop()
        video_path = self.video.stop()
        for t in self.threads:
            t.join()

        return {
            "audio_dir": audio_path,
            "video_dir": video_path,
            "audio_summary": self.audio.get_summary(),
            "video_summary": self.video.get_summary()
        }

global_av_controller = None


def start_av_recording() -> str:
    """启动音视频采集"""
    global global_av_controller
    global_av_controller = AVController()
    global_av_controller.start()
    return "🎙️🎥 正在采集音视频..."

def stop_av_recording() -> dict:
    print("🟥 stop_record 节点被触发")
    """停止音视频采集并返回分析结果摘要"""
    global global_av_controller
    if global_av_controller:
        result = global_av_controller.stop()
        return result
    else:
        return {
            "error": "当前没有正在采集的任务"
        }
