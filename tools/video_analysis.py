import cv2
import threading
import queue
import time
import json
from pathlib import Path
from datetime import datetime
import sys

sys.path.append(str(Path(__file__).resolve().parent.parent))
from agents.video_agent import build_video_graph  # 你已有的分析函数

class VideoController:
    def __init__(self):
        self.exit_flag = threading.Event()
        self.video_queue = queue.Queue()
        self.video_threads = []
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir = Path(f"output/{self.timestamp}/video")
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def video_stream_worker(self):
        cap = cv2.VideoCapture(0)
        idx = 0
        frame_buffer = []
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 20.0
        frame_target = int(fps * 5)  # 5秒一个片段

        def save_video_callback(frames, idx):
            video_path = self.output_dir / f"video_{idx}.mp4"
            out = cv2.VideoWriter(str(video_path), fourcc, fps, (width, height))
            for f in frames:
                out.write(f)
            out.release()
            self.video_queue.put(str(video_path))
            print(f"🎬 保存视频段 {idx}: {video_path}")

        print("[视频线程] 启动")
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_buffer.append(frame)
            cv2.imshow("🎥", frame)

            # ✅ 每一帧都判断是否被终止
            if self.exit_flag.is_set():
                if frame_buffer:
                    save_video_callback(frame_buffer, idx)
                break

            if len(frame_buffer) >= frame_target:
                save_video_callback(frame_buffer, idx)
                frame_buffer.clear()
                idx += 1

            if cv2.waitKey(1) & 0xFF == ord('q'):
                self.exit_flag.set()
                break

        cap.release()
        cv2.destroyAllWindows()
        self.video_queue.put("DONE")
        print("[视频线程] 结束")

    def video_analysis_worker(self):
        print("[分析线程] 启动")
        graph = build_video_graph()
        results = []

        while True:
            try:
                video_path = self.video_queue.get(timeout=1)
                if video_path == "DONE":
                    print("[分析线程] 收到结束标志")
                    break

                print(f"[分析线程] 分析视频：{video_path}")
                result = graph.invoke({"video_path": video_path})
                results.append({
                    "video_path": video_path,
                    "video_analysis": result
                })
            except queue.Empty:
                if self.exit_flag.is_set():
                    break
                continue

        with open(self.output_dir / "video_analysis.json", "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

        print("[分析线程] 结束")

    def start(self):
        self.exit_flag.clear()
        self.video_threads = []
        t1 = threading.Thread(target=self.video_stream_worker)
        t2 = threading.Thread(target=self.video_analysis_worker)
        t1.start()
        t2.start()
        self.video_threads.extend([t1, t2])

    def stop(self):
        self.exit_flag.set()
        for t in self.video_threads:
            t.join()
        print("✅ 视频录制和分析结束")
        return str(self.output_dir)
    
    def get_summary(self) -> str:
        analysis_path = self.output_dir / "video_analysis.json"
        if not analysis_path.exists():
            return "未找到视频分析结果"

        with open(analysis_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        summary_lines = []
        for d in data:
            summary = d["video_analysis"].get("video_analysis", "无表情分析")
            video_path = d.get("video_path", "")
            summary_lines.append(f"🎥 {video_path}: {summary}")

        return "\n".join(summary_lines)

    
