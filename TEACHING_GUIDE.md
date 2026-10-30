# 📚 DermaGuard Teaching Guide
### "What Are We Doing, How, and What Tools Are We Using?"

> This file exists because **Dhruv has zero prior knowledge** of machine learning, Python libraries,
> and production software engineering. Every time we add a new feature or tool, this file gets
> updated with a plain-English explanation of what it is and why we chose it.
> **Easy language is mandatory here. No jargon without an explanation right after it.**

---

## 🧠 The Big Picture in One Sentence

> We are building a **smart AI system** that looks at close-up photos of skin spots, figures out
> if the photo is good enough to analyze, then tries to identify what kind of skin condition it is,
> and finally tells a doctor or patient what to do next.

---

## 🏗️ The Full-Split Architecture: Backend vs Frontend

To make our software **production-grade**, we divided the project into two independent parts:

### 1. The Backend (`backend/`) — The Brain & Muscle
* **Who works here**: You and the AI.
* **What it contains**: 
  - The Image Quality Agent (OpenCV code that checks blur, glare, and darkness).
  - The Diagnosis Agent (PyTorch AI model that recognizes the 7 skin diseases).
  - The Triage Engine (the medical rules that output `AUTO_TRIAGE`, `REFER`, or `RE_IMAGE`).
  - The REST API Server (`api_server.py`) that answers questions from the frontend.

### 2. The Frontend (`frontend/`) — The Face & User Experience
* **Who works here**: Your teammate/friend.
* **What it contains**: 
  - The user interface (webpage with buttons, file uploaders, and colorful badges).
  - Works in any browser by opening `index.html`.
* **How it talks to the Backend**: It sends an HTTP POST request to `http://localhost:8000/api/triage` with the photo, and receives back a clean JSON answer.

### Why is this called "Production Grade"?
In professional software companies (Google, Meta, Apple), you never mix user interface code directly with machine learning math. If the frontend crashes, the AI is unaffected. If the AI is being updated, the frontend still works. Furthermore, two developers can work at the exact same second without ever having Git conflicts!

---

## 🔧 Tools We Are Using (And What They Do)

### 1. Python 🐍
**What it is**: A programming language. Think of it as the language we use to write instructions for the computer.
**Why we chose it**: It's the industry standard for AI and machine learning.
**Analogy**: If a recipe book is the program, Python is the language the recipe is written in.

---

### 2. PyTorch 🔥
**What it is**: A library (a collection of pre-written code) that makes building AI models easy.
**What it does for us**: It trains our skin-disease classifier and does millions of math calculations per second.
**Key concept — MPS (Metal Performance Shaders)**:
Your MacBook has an Apple Silicon M-chip. PyTorch uses MPS to run math on your Mac's graphics chip, making training 5–10x faster than normal CPU.

---

### 3. OpenCV (cv2) 👁️
**What it is**: "Open Computer Vision" — A library for analyzing images using code.
**What it does for us**: Powers all 4 checks in our **Image Quality Agent** (blur detection, glare detection, lighting, and dark border obstruction).
**Key concept — Laplacian Variance**:
A mathematical filter that finds sharp edges. High number = crisp photo. Low number = blurry photo.

---

### 4. Torchvision 🖼️
**What it is**: An add-on for PyTorch specifically for photos and computer vision.
**What it does for us**: Provides the **MobileNetV3** neural network and tools to resize and normalize photos.
**Key concept — Transfer Learning**:
Starting with an AI that already knows how to see shapes and colors from millions of internet photos, and just teaching it the last step: distinguishing the 7 skin lesion types.

---

### 5. Scikit-learn 📊
**What it is**: The gold-standard library for measuring how well an AI performs.
**What it does for us**: Calculates our Accuracy, Balanced Accuracy, F1 Score, and Confusion Matrix across the 2,000 validation images.

---

### 6. Pydantic ✅
**What it is**: A library that enforces strict data shapes in Python.
**What it does for us**: Makes sure that data moving between agents is never missing a field or formatted incorrectly.

---

### 7. REST API (Representational State Transfer Application Programming Interface) 🌐
**What it is**: The standard protocol for computers to talk to each other over the web or locally.
**What it does for us**: When your friend's frontend sends a photo to `http://localhost:8000/api/triage`, our API server processes it and replies with clean JSON.

---

### 8. Pytest 🧪
**What it is**: An automated testing robot.
**What it does for us**: Runs 19 automated tests in 5 seconds to verify that every single quality rule, model inference, and triage decision works properly.

---

## 📋 How to Run the Project (Step by Step)

### 1. Start the Backend API Server:
```bash
cd backend
../.venv/bin/python -m src.api_server
```

### 2. Open the Frontend UI:
Simply open `frontend/index.html` in Safari, Chrome, or any browser! Drag and drop a skin image, and watch the multi-agent pipeline assess it live!

### 3. Run the Automated Tests:
```bash
cd backend
../.venv/bin/pytest -v
```
All 19 tests will pass in green!
