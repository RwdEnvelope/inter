# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.
必须用中文回复

## Project Overview

This is an AI-powered interview system that synchronously captures and analyzes audio and video streams during interview sessions. The system uses LangGraph workflows to orchestrate the interview process, combining real-time audio/video recording with AI analysis.

## Key Architecture Components

### Core Workflow (`graph/`)

- **graph.py**: Main interview orchestration using LangGraph StateGraph. Manages interview rounds, generates questions via LLM, and coordinates audio/video capture
- **av_workflow.py**: Subgraph for audio/video recording workflow with time-based controls and stop signals

### Agent System (`agents/`)

- **question_agent.py**: Creates structured LLM agent for generating interview questions based on candidate resume and conversation history
- **audio_agent.py**: Processes audio transcription and emotion analysis
- **video_agent.py**: Handles video analysis for facial expressions and body language

### Recording & Analysis (`tools/`)

- **analysis.py**: Global controller that coordinates audio and video recording threads
- **audio_analysis.py**: RecorderController class managing real-time audio capture, chunking (5-second segments), and transcription pipeline
- **video_analysis.py**: VideoController class handling camera capture, video segmentation, and facial analysis

### Data Flow

1. Interview starts → LLM generates questions → av_interview_tool triggered
2. Parallel recording: Audio chunks (5s) + Video segments (5s) saved to timestamped output directories
3. Real-time analysis: Audio transcription + emotion detection, Video facial expression analysis
4. Results aggregated and fed back to LLM for next question generation
5. Process continues until max rounds reached or LLM decides to end

## Running the System

### Start Interview Session

```bash
python graph/graph.py
```

### Launch Gradio Interface

```bash
python test_agent.py
```

### Development Testing

```bash
python graph/av_workflow.py  # Test AV workflow separately
```

## Dependencies

- LangGraph for workflow orchestration
- LangChain for LLM integration (OpenAI GPT-4o)
- sounddevice + soundfile for audio recording
- OpenCV for video capture
- Gradio for web interface

## Output Structure

```
output/
  YYYYMMDD_HHMMSS/
    audio/
      audio_0.wav, audio_1.wav, ...
      transcripts.json
      audio_analysis.json
    video/
      video_0.mp4, video_1.mp4, ...
      video_analysis.json
```

## State Management

The system uses TypedDict states:

- **InterviewState**: Tracks interview rounds, Q&A pairs, audio/video summaries
- **AVState**: Manages recording status and stop signals in the AV subgraph

## Important Implementation Notes

- Audio/video recording runs in separate threads with queue-based communication
- The system supports external stop signals and timeout controls
- All file paths use timestamp-based organization for session isolation
- LLM integration requires OpenAI API configuration
