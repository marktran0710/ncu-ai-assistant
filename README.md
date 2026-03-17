# 🤖 Gemini RAG AI Agent

A simple **Retrieval-Augmented Generation (RAG)** agent built with **LangChain** and **Google Gemini**. This project allows you to query PDF documents using Google's state-of-the-art embedding and generative models.

---

## 🛠️ Setup Instructions

Follow these steps to set up and run the project in an isolated Python environment.

### 1. Create a Virtual Environment

Isolate your project dependencies to avoid version conflicts.

- **Windows:**
  ```bash
  python -m venv .venv
  ```
- **macOS / Linux:**
  ```bash
  python3 -m venv .venv
  ```

### 2. Activate the Environment

Activate the environment **every time** you open a new terminal session for this project.

- **Windows (Command Prompt):**
  ```cmd
  .venv\Scripts\activate
  ```
- **Windows (PowerShell):**
  ```powershell
  .\.venv\Scripts\Activate.ps1
  ```
- **macOS / Linux:**
  ```bash
  source .venv/bin/activate
  ```

> **Tip:** You’ll know it’s active when you see `(.venv)` at the start of your terminal prompt.

### 3. Install Dependencies

With the virtual environment active, install the required packages from `requirements.txt`.

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

> 💡 If you want to verify where a package is installed (system vs `.venv`):
>
> ```bash
> python -m pip show <package-name>
> ```
>
> Look at the `Location:` field; it should point inside `.venv/` when the venv is active.

### 4. Run the Agent

Once dependencies are installed and the venv is active, run:

```bash
python main.py
```

---

## ✅ Quick Checks

- Ensure Python comes from the venv:
  ```bash
  python -c "import sys; print(sys.executable)"
  ```
- List installed packages in the active environment:
  ```bash
  pip list
  ```
