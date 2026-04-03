# ⚡ KaggleClaw

KaggleClaw is an elite, autonomous machine learning agent built to win Kaggle competitions. It leverages the power of massive LLMs (like OSS-120B) and a suite of specialized tools to perform EDA, train models, and optimize submissions entirely on its own.

## 🏗️ Architecture

KaggleClaw has been refactored into a modular, scalable pipeline:

- **`agent/core/`**: The brains of the operation.
    - `runner.py`: Orchestrates the agentic loop and tool dispatching.
    - `client.py`: Handles structured communication with vLLM using Harmony.
    - `events.py`: Unified SSE event system.
- **`api/`**: FastAPI server providing a web interface and real-time streaming.
- **`config/`**: Centralized configuration management via `settings.py`.
- **`agent/tools/`**: Extensible toolset (Python, Browser, File Management, etc.).
- **`frontend/`**: Modern, responsive UI for monitoring agent progress.

## 🚀 Getting Started

1. **Environment Setup**:
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   pip install -r req.txt
   ```

2. **Configuration**:
   Edit `env.py` or set environment variables:
   - `VLLM_BASE_URL`: URL of your vLLM server.
   - `NGROK_AUTH_TOKEN`: Your ngrok token for public access in Kaggle.

3. **Run**:
   ```bash
   python main.py
   ```

## 🧠 Design Philosophy

- **Autonomy**: High reasoning effort, iterative debugging, and self-correction.
- **Modularity**: Easy to add new tools and swapping LLM backends.
- **Visibility**: Real-time streaming of thinking process and tool executions.

---
*Built with ❤️ for Kaggle competitors.*
