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

## 🩺 The Problem We're Solving

Imagine a patient takes a photo of a suspicious skin mole on their phone and sends it to a clinic.
Three bad things can happen with a basic AI:

1. **Photo is blurry** — AI still guesses a diagnosis. That guess could be wrong and dangerous.
2. **AI says it's a harmless mole** — but it's actually melanoma (skin cancer). Patient goes home. They die.
3. **AI says it's cancer** — it's actually harmless. Patient panics and gets unnecessary surgery.

**Our solution**: Instead of one AI doing everything blindly, we have **three specialist agents** that
check each other's work, just like a real medical team.

---

## 🔧 Tools We Are Using (And What They Do)

### 1. Python 🐍
**What it is**: A programming language. Think of it as the language we use to write instructions for the computer.
**Why we chose it**: It's the industry standard for AI and machine learning. Almost every AI company in the world uses Python.
**Analogy**: If a recipe book is the program, Python is the language the recipe is written in.

---

### 2. PyTorch 🔥
**What it is**: A library (a collection of pre-written code) that makes building AI models easy.
**What it does for us**: It trains our skin-disease classifier. It handles all the complex math (millions of calculations per second) so we don't have to write it ourselves.
**Why we chose it**: It's made by Meta (Facebook) and used by nearly every AI research lab in the world. It supports Apple Silicon chips (your MacBook's M-chip) through something called **MPS**.
**Analogy**: PyTorch is like a well-equipped kitchen. Instead of building your own oven, you just use the one that's already there.

**Key concept — MPS (Metal Performance Shaders)**:
Your MacBook has an Apple Silicon chip (M1/M2/M3). It has a built-in "GPU" (Graphics Processing Unit — a chip originally for games but extremely fast at AI math). MPS lets PyTorch use that chip, making training 5–10x faster than using your normal CPU.

---

### 3. OpenCV (cv2) 👁️
**What it is**: "Open Computer Vision" — A library for analyzing and manipulating images using code.
**What it does for us**: Powers all 4 checks in our **Image Quality Agent** (blur detection, glare detection, etc.).
**Why we chose it**: The standard library for computer vision tasks. Used in everything from self-driving cars to face recognition.
**Analogy**: If Python is the language and PyTorch is the kitchen, OpenCV is the specialized set of knives and measuring tools for examining photos.

**Key concept — Laplacian Variance**:
The Laplacian is a mathematical filter that highlights edges (where dark meets bright) in an image.
A sharp photo has lots of strong, clear edges → high variance number.
A blurry photo has soft, smudged edges → low variance number.
We use this number to decide if a photo is sharp enough for medical diagnosis.

---

### 4. Torchvision 🖼️
**What it is**: An add-on for PyTorch specifically for image-related tasks.
**What it does for us**: Provides pre-trained image classification models (MobileNetV3) and image transformation tools (resizing, normalizing colors).
**Why we chose it**: Comes with models already trained on millions of images from the internet, giving us a huge head start.

**Key concept — Transfer Learning**:
Instead of teaching our AI from scratch (which would need millions of skin images and weeks of training),
we start with a model that already knows how to recognize shapes, colors, textures, and edges from internet images.
Then we just teach it the *last step*: "When you see these patterns, it's melanoma. When you see those patterns, it's a harmless mole."
This is **Transfer Learning** — transferring knowledge from one problem to another.
**Analogy**: Hiring a chef who already knows knife skills and cooking techniques, and just training them on your specific restaurant's menu instead of starting from zero.

---

### 5. Scikit-learn 📊
**What it is**: A library for machine learning utility tools — especially for measuring how good an AI is.
**What it does for us**: Calculates our performance metrics after running on 2,000 validation images.
**Why we chose it**: The gold standard for ML evaluation. Battle-tested and trusted.

**Key concept — Metrics we calculate**:
- **Accuracy**: Out of 2,000 images, what percentage did we classify correctly? (e.g., 78%)
- **Balanced Accuracy**: Like accuracy, but adjusted to be fair when some classes appear much more than others (our dataset has 67% moles and only 5% melanoma — balanced accuracy corrects for this).
- **F1 Score**: A combined measure of "Did we catch all the sick cases?" and "Did we avoid false alarms?". Critical in medicine.
- **Confusion Matrix**: A 7×7 table showing *which* diseases we confused with each other. E.g., did we mistake melanoma for a mole?

---

### 6. Pydantic ✅
**What it is**: A library that enforces strict data structures in Python.
**What it does for us**: Defines the exact "shape" of data that flows between our agents.
**Why we chose it**: Prevents bugs where one agent sends data in a format that another agent can't understand.
**Analogy**: Pydantic is like a standardized form. Every agent fills in the exact same form, so no information is lost or mis-formatted when data passes between them.

---

### 7. Pandas 🐼
**What it is**: A library for working with tables and spreadsheets in Python.
**What it does for us**: Reads the `metadata CSV` file that maps each image filename to its disease label.
**Why we chose it**: The most popular data manipulation library in Python. Reading a CSV is one line of code.

---

### 8. Pytest 🧪
**What it is**: A library for automated testing.
**What it does for us**: Runs 18 automated checks every time we make a change, to make sure we didn't break anything.
**Why we chose it**: In production software, you never manually test every feature after every change. You write automated tests that do it for you in seconds.
**Analogy**: A car factory has robots that check every bolt and joint before the car leaves the factory. Pytest is our robot inspector.

---

### 9. tqdm 📊
**What it is**: A tiny library that shows a progress bar in the terminal.
**What it does for us**: Shows a progress bar while processing 2,000 validation images so we know how far along we are.
**Why we chose it**: Processing 2,000 images takes a few minutes. Without a progress bar, you'd have no idea if the program is running or crashed.

---

## 🏗️ Key Software Engineering Concepts We Are Using

### What is a "Library" or "Package"?
Pre-written code that someone else built, tested, and published for everyone to use.
Instead of writing blur detection from scratch (thousands of lines), we `import cv2` and use their work.
All our libraries are listed in `requirements.txt`.

### What is a Virtual Environment (`.venv`)?
A self-contained, isolated copy of Python just for our project.
**Why we need it**: Different projects need different versions of libraries. A virtual environment means our project's libraries don't interfere with other projects on your Mac.
**Analogy**: Like having a separate toolbox for each project instead of throwing all tools in one box.

### What is a "Module" or "Package" in Python?
A Python file (`.py`) = a **module**. A folder of Python files = a **package**.
Our `src/` folder is a package. Each `.py` file inside is a module.
`__init__.py` files (the empty-looking ones) tell Python "this folder is a package, not just a random folder."

### What is an "Agent" in Multi-Agent AI?
In AI, an **agent** is an autonomous program that:
1. Takes in some input (like an image)
2. Performs a specific task
3. Produces a specific output

We have 3 agents. Each is independent and only does its job. They then hand off results to the next agent.
**This is like an assembly line**: photographer → quality inspector → doctor → triage nurse.

### What is a "Base Class"?
Think of it as a **contract** or **blueprint**. Our `BaseAgent` class says:
"Every agent you build MUST have a `process()` method."
This ensures all agents are built the same way, making the system consistent and predictable.
**Analogy**: A hospital policy that says every department must use the same patient intake form.

### What is a "Schema" (Pydantic Model)?
A defined structure for data. Our `ImageQualityResult` schema says:
- Must have a `quality_status` field (PASS/WARNING/FAIL)
- Must have a `quality_score` between 0.0 and 1.0
- Must have a `reasons` list
If any field is missing or wrong type, Pydantic throws an error immediately.

### What is "Type Safety" / "Type Annotations"?
In Python, `def process(img: np.ndarray) -> ImageQualityResult:` tells us and the tools:
- `img` must be a NumPy array
- The function will return an `ImageQualityResult`
This catches bugs early, before they cause damage in production.

---

## 🔄 How Data Flows Through Our System

```
1. You give us an image file (e.g., ISIC_0024306.jpg)
         ↓
2. Image Quality Agent inspects the raw pixels with OpenCV
   → Checks blur (Laplacian), glare (bright pixel %), lighting (mean brightness), border obstruction
   → Outputs: ImageQualityResult { status: PASS/FAIL, score: 0.85, reasons: [...] }
         ↓
3. Diagnosis Agent loads the image, resizes it to 224×224, normalizes colors,
   then passes it through MobileNetV3 neural network
   → Outputs: DiagnosisResult { predicted_class: "nv", confidence: 0.91, probabilities: {...} }
         ↓
4. Triage Engine reads BOTH results and applies clinical rules
   → Rule 1: Quality FAIL? → RE_IMAGE (stop, don't diagnose bad photos)
   → Rule 2: Melanoma/BCC/AKIEC? → REFER (send to dermatologist)
   → Rule 3: Confidence < 70%? → REFER (uncertain, be safe)
   → Rule 4: All good? → AUTO_TRIAGE (benign, clear for routine care)
   → Outputs: FinalTriageResult { decision: REFER, reasons: [...] }
         ↓
5. Reporter compiles all 2,000 results into one JSON payload for API submission
```

---

## 📋 How to Run the Project (Step by Step for Beginners)

### Step 1: Activate the virtual environment
```bash
cd /Users/dhruvkumarlodhi/Documents/Hackathon/team_pirates
source .venv/bin/activate
```
You'll see `(.venv)` appear in your terminal. This means you're inside the project's isolated Python environment.

### Step 2: Run the automated tests (to verify everything is working)
```bash
pytest -v
```
All 18 tests should say `PASSED` in green.

### Step 3: Train the diagnosis model (needs dataset first!)
```bash
python -m src.cli train \
    --images data/train \
    --metadata data/train_metadata.csv \
    --epochs 10
```

### Step 4: Run the full validation pipeline
```bash
python -m src.cli validate \
    --val-dir data/val \
    --ground-truth data/val_metadata.csv \
    --output-json reports/validation_submission_payload.json
```

### Step 5: Test quality check on a single image
```bash
python -m src.cli assess-quality --image path/to/your/image.jpg
```

---

*This file is maintained by the AI assistant and updated every time a new tool, concept, or feature is added to the project.*
