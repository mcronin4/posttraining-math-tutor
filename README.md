# LLM Math Tutor

A monorepo for building, training, and deploying an AI-powered math tutoring assistant aligned with Ontario K–12 curriculum.

## Architecture

```
├── apps/
│   ├── web/          # Next.js frontend
│   └── api/          # FastAPI backend
├── packages/
│   ├── core/         # Shared types, schemas, prompt templates
│   ├── data/         # Data processing and taxonomy tools
│   ├── eval/         # Evaluation harness
│   └── training/     # Fine-tuning scripts
```

## Project Focus

**Primary Goal:** Benchmark and improve math tutoring models through fine-tuning.

This project focuses on:
1. **Benchmarking** baseline models (qwen3:8b) against multiple evaluation suites
2. **Training** improved variants using Tinker (SFT, Distillation, RLVR)
3. **Comparing** performance improvements across training methods
4. **Tracking** experiments and results systematically

The frontend/API are primarily for testing and demonstration. The core work is in:
- `packages/eval/` - Comprehensive benchmarking suite ([README](packages/eval/README.md))
- `packages/training/` - Training scripts for different methods ([README](packages/training/README.md))
- `packages/data/` - Data processing and tagging tools ([README](packages/data/README.md))
- `WORKFLOW.md` - Complete training and benchmarking workflow

**Documentation:**
- [WORKFLOW.md](WORKFLOW.md) - Complete training and benchmarking guide
- [CONTRIBUTING.md](CONTRIBUTING.md) - Development guidelines and codebase overview
- Package READMEs - See each package directory for detailed documentation

## Quick Start

```bash
# Prerequisites: Node.js 18+, Python 3.11+, Ollama with qwen3:8b model

# Install dependencies
npm install
make install  # or run uv sync in each package manually

# Start services (in separate terminals)
make api-dev  # Terminal 1: Backend on :8000
make web-dev  # Terminal 2: Frontend on :3000

# Open http://localhost:3000
```

---

## Complete Setup Guide

This guide will walk you through installing everything needed to run the project

### Step 1: Install Node.js

**Windows:**
1. Go to https://nodejs.org/
2. Download the **LTS version** (recommended, e.g., v20.x.x)
3. Run the installer (.msi file)
4. Follow the installation wizard (accept all defaults)
5. **Restart your terminal/command prompt** after installation

**Verify installation:**
```bash
node --version
# Should show: v20.x.x or higher

npm --version
# Should show: 10.x.x or higher
```

---

### Step 2: Install Python

**Windows:**
1. Go to https://www.python.org/downloads/
2. Download **Python 3.11 or newer** (e.g., Python 3.12)
3. Run the installer
4. **IMPORTANT:** Check the box "Add Python to PATH" during installation
5. Click "Install Now"

**Verify installation:**
```bash
python --version
# Should show: Python 3.11.x or higher
```

---

### Step 3: Install uv (Python Package Manager)

**Windows (PowerShell as Administrator):**
1. Open PowerShell as Administrator:
   - Press `Win + X`
   - Click "Windows PowerShell (Admin)" or "Terminal (Admin)"
   - Click "Yes" if prompted

2. Run this command:
   ```powershell
   powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
   ```

3. **Close and reopen** your terminal/PowerShell

**Alternative (if above doesn't work):**
```bash
pip install uv
```

**Verify installation:**
```bash
uv --version
# OR if that doesn't work:
python -m uv --version
```

**Note:** If `uv` command isn't found, you can always use `python -m uv` instead throughout this guide.

---

### Step 4: Install Ollama

**What it is:** Ollama runs large language models (like Qwen) locally on your computer.

**Windows:**
1. Go to https://ollama.ai
2. Click "Download" for Windows
3. Run the installer (`OllamaSetup.exe`)
4. Follow the installation wizard
5. Ollama will start automatically after installation

**Verify installation:**
- You should see the Ollama app icon in your system tray (bottom-right corner)
- Open the Ollama app to see the interface

**Note:** The `ollama` command-line tool may not be available in Git Bash. You can:
- Use the Ollama app UI instead, OR
- Use PowerShell/Command Prompt for `ollama` commands

---

### Step 5: Download a Model in Ollama

**What it is:** The model (Qwen 3 8B) is the AI brain that generates tutoring responses.

**Option A: Using Ollama App (Easiest)**
1. Open the Ollama app
2. Click on the "Models" or "Library" tab
3. Search for "qwen3"
4. Find "qwen3:8b" and click "Pull" or "Download"
5. Wait for download to complete (~5GB, may take 5-10 minutes)

**Option B: Using Command Line (PowerShell)**
```powershell
ollama pull qwen3:8b
```

**Verify model is downloaded:**
- In Ollama app, check the Models tab - you should see `qwen3:8b` listed
- Or in PowerShell: `ollama list` (if command works)

---

### Step 6: Install Make (Windows)

**Windows (using Chocolatey):**
1. Open PowerShell as Administrator (see Step 3 for how)
2. Install Chocolatey (if not already installed):
   ```powershell
   Set-ExecutionPolicy Bypass -Scope Process -Force; [System.Net.ServicePointManager]::SecurityProtocol = [System.Net.ServicePointManager]::SecurityProtocol -bor 3072; iex ((New-Object System.Net.WebClient).DownloadString('https://community.chocolatey.org/install.ps1'))
   ```
3. Install make:
   ```powershell
   choco install make -y
   ```
4. **Close and reopen** your terminal

**Verify installation:**
```bash
make --version
# Should show: GNU Make 4.x
```

**Alternative:** If you don't want to install make, you can run commands directly (see "Running Without Make" section below).

---

### Step 7: Clone and Setup the Project

**1. Clone the repository** (if you haven't already):
```bash
git clone <repository-url>
cd posttraining-math-tutor
```

**2. Install Node.js dependencies:**
```bash
npm install
```
This installs packages needed for the web frontend.

**3. Install Python dependencies:**

For each package, run:
```bash
# Backend API
cd apps/api
python -m uv sync
cd ../..

# Data processing tools
cd packages/data
python -m uv sync
cd ../..

# Evaluation scripts
cd packages/eval
python -m uv sync
cd ../..

# Training scripts
cd packages/training
python -m uv sync
cd ../..
```

**Or use make** (if installed):
```bash
make install
```

**4. Configure environment variables:**

Create `.env` files from examples:
```bash
# For the API
cp apps/api/env.example apps/api/.env

# For the web app (optional)
cp apps/web/env.example apps/web/.env.local
```

Edit `apps/api/.env` if needed (defaults should work):
```env
OLLAMA_MODEL=qwen3:8b
OLLAMA_BASE_URL=http://localhost:11434
```

---

### Step 8: Verify Everything Works

**1. Make sure Ollama is running:**
- Check system tray for Ollama icon
- Or open the Ollama app

**2. Start the API backend** (Terminal 1):
```bash
make api-dev
# OR without make:
cd apps/api && python -m uv run uvicorn src.main:app --reload --port 8000
```

You should see:
```
🚀 Starting Math Tutor API...
📚 Using model adapter: OllamaAdapter
INFO:     Uvicorn running on http://0.0.0.0:8000
```

**3. Start the web frontend** (Terminal 2):
```bash
make web-dev
# OR without make:
cd apps/web && npm run dev
```

You should see:
```
▲ Next.js 14.1.0
- Local:        http://localhost:3000
```

**4. Open your browser:**
- Go to http://localhost:3000
- You should see the Math Tutor interface!

**5. Test it:**
- Type a math question like "What is 2 + 3?"
- Select a mode (Hint, Check Step, or Explain)
- Click send
- You should get a response from the AI tutor!

---

## Running Without Make

If you don't have `make` installed, here are the direct commands:

**Start API:**
```bash
cd apps/api
python -m uv run uvicorn src.main:app --reload --port 8000
```

**Start Web Frontend:**
```bash
cd apps/web
npm run dev
```

**Run Evaluations:**
```bash
cd packages/eval
python -m uv run python eval_math.py --dataset samples/math_samples.jsonl --endpoint http://localhost:8000/chat
```

---

## Development Workflow

Once everything is set up, your typical workflow:

1. **Start Ollama** (if not already running)
2. **Terminal 1:** Run `make api-dev` (backend)
3. **Terminal 2:** Run `make web-dev` (frontend)
4. **Browser:** Open http://localhost:3000
5. **Code:** Make changes, servers auto-reload
6. **Test:** Try questions in the UI

To stop servers: Press `Ctrl+C` in each terminal.

### Evaluation

**Prerequisites:** Make sure the API is running (`make api-dev`)

```bash
# Run all smoke tests
make eval-smoke

# Run individual evaluations
make eval-math      # Math accuracy evaluation
make eval-tutor     # Tutoring quality evaluation
make eval-safety    # Safety/refusal evaluation
```

Results are saved to `packages/eval/outputs/` as JSON files.

### Training (Skeleton)

**Note:** The training pipeline is currently a skeleton. To implement:

```bash
# Run SFT training (requires actual implementation)
make train-sft
```

See `packages/training/sft.py` for the training script structure.

## Configurations

Copy `.env.example` files in each app/package to `.env` and configure as needed:

```bash
cp apps/web/.env.example apps/web/.env.local
cp apps/api/.env.example apps/api/.env
```

## API Endpoints

### POST /chat

Request:
```json
{
  "question": "What is 2 + 3?",
  "attempt": "I think it's 5",
  "mode": "hint|check_step|explain",
  "grade": "K|1|2|...|12",
  "dont_reveal_answer": true,
  "topic_tags": ["addition", "arithmetic"]
}
```

Response:
```json
{
  "response": "Great job! You're on the right track...",
  "refusal": false,
  "citations": [],
  "debug": {
    "selected_policy": "validate_correct"
  }
}
```

## Tutoring Modes

| Mode | Description |
|------|-------------|
| **Hint** | Provide guiding questions to help student discover the answer |
| **Check Step** | Validate student's work and suggest next steps |
| **Explain** | Explain the concept without giving away the final answer |

## License

MIT

