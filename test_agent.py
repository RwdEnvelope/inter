# gradio_av.py

import gradio as gr
from tools.analysis import start_av_recording, stop_av_recording

# === 页面函数 ===
def gr_start():
    status = start_av_recording()
    return status

def gr_stop():
    result = stop_av_recording()
    return result

# === 构建 Gradio 页面 ===
with gr.Blocks(title="音视频同步采集系统") as demo:
    gr.Markdown("## 🎙️🎥 音视频同步采集系统")

    with gr.Row():
        start_btn = gr.Button("▶️ 开始采集")
        stop_btn = gr.Button("⏹️ 停止采集")

    status_output = gr.Textbox(label="状态 / 输出路径", lines=4)

    start_btn.click(fn=gr_start, outputs=status_output)
    stop_btn.click(fn=gr_stop, outputs=status_output)

# === 启动页面 ===
if __name__ == "__main__":
    demo.launch()

