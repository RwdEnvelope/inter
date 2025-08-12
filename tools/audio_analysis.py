# audio_analysis.py
import sounddevice as sd
import soundfile as sf
import numpy as np
import threading
import queue
import time
import json
from pathlib import Path
from datetime import datetime
import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))

from agents.audio_agent import build_audio_graph

class RecorderController:
    def __init__(self):
        self.exit_flag = threading.Event()
        self.audio_queue = queue.Queue()
        self.recording_threads = []
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir = Path(f"output/{self.timestamp}/audio")
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def audio_stream_worker(self):
        AUDIO_SR = 16000
        CHUNK_DURATION = 5
        AUDIO_FRAME_COUNT = AUDIO_SR * CHUNK_DURATION
        idx = 0
        current_audio = []
        accumulated_frames = 0

        def callback(indata, frames, time_info, status):
            nonlocal current_audio, idx, accumulated_frames
            current_audio.append(indata.copy())
            accumulated_frames += frames

            if accumulated_frames >= AUDIO_FRAME_COUNT:
                audio_chunk = np.concatenate(current_audio, axis=0)
                audio_path = self.output_dir / f"audio_{idx}.wav"
                sf.write(str(audio_path), audio_chunk, AUDIO_SR)
                self.audio_queue.put(str(audio_path))
                print(f"[录音线程] 保存音频：{audio_path}")
                current_audio.clear()
                accumulated_frames = 0
                idx += 1

        print("[录音线程] 启动")
        with sd.InputStream(samplerate=AUDIO_SR, channels=1, callback=callback):
            while not self.exit_flag.is_set():
                time.sleep(0.1)
        print("[录音线程] 停止")
        self.audio_queue.put("DONE")

    def audio_analysis_worker(self):
        print("[分析线程] 启动")
        graph = build_audio_graph()
        transcripts = []
        analyses = []

        while True:
            try:
                audio_path = self.audio_queue.get(timeout=1)
                if audio_path == "DONE":
                    print("[分析线程] 收到结束标志")
                    break

                print(f"[分析线程] 处理：{audio_path}")
                result = graph.invoke({"audio_path": audio_path})
                transcripts.append({
                    "audio_path": audio_path,
                    "transcript": result.get("transcript", "")
                })
                analyses.append({
                    "audio_path": audio_path,
                    "audio_analysis": result.get("audio_analysis", {})
                })
            except queue.Empty:
                if self.exit_flag.is_set():
                    break
                else:
                    continue

        with open(self.output_dir / "transcripts.json", "w", encoding="utf-8") as f:
            json.dump(transcripts, f, ensure_ascii=False, indent=2)
        with open(self.output_dir / "audio_analysis.json", "w", encoding="utf-8") as f:
            json.dump(analyses, f, ensure_ascii=False, indent=2)

        print("[分析线程] 结束")

    def start(self):
        self.exit_flag.clear()
        self.recording_threads = []
        t1 = threading.Thread(target=self.audio_stream_worker)
        t2 = threading.Thread(target=self.audio_analysis_worker)
        t1.start()
        t2.start()
        self.recording_threads.extend([t1, t2])

    def stop(self):
        self.exit_flag.set()
        for t in self.recording_threads:
            t.join()
        print("✅ 所有线程结束")
        return str(self.output_dir)
    def get_summary(self) -> str:
        """把全部 transcript 拼成一段；把全部 emotion 拼成一段"""

        transcript_path = self.output_dir / "transcripts.json"
        analysis_path   = self.output_dir / "audio_analysis.json"

        if not transcript_path.exists():
            return "未找到语音转录结果"

        # 1. 读取文件
        with open(transcript_path, "r", encoding="utf-8") as f:
            transcripts = json.load(f)
        with open(analysis_path, "r", encoding="utf-8") as f:
            analyses = json.load(f)

        # 2. 拼接所有文字
        full_text = " ".join(
            t.get("transcript", "").strip() for t in transcripts if t.get("transcript", "").strip()
        )

        emotion_items = []
        for a in analyses:
            emo = a.get("audio_analysis", {})
            if isinstance(emo, dict):
                emotion_items.extend(f"{k}:{v}" for k, v in emo.items())
            elif emo:
                emotion_items.append(str(emo))

        emotion_summary = "；".join(emotion_items) if emotion_items else "无"

    # 4. 返回两段式摘要
        return f"语音内容: {full_text}\n情绪分析: {emotion_summary}"


