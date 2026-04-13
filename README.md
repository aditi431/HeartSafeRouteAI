# 🫀 HeartSafe Route AI  
### AI-Powered Emergency Navigation System for Heart Patients (OpenEnv RL Environment)

---

## 🚨 Overview

HeartSafe Route AI is an **OpenEnv-compliant Reinforcement Learning environment** where AI agents learn to make life-saving decisions by selecting the **safest and fastest cardiac emergency routes** under real-world constraints such as:

- 🚦 Traffic conditions  
- 🫁 Air pollution levels  
- 🏥 Hospital capacity & availability  
- ⏱️ Emergency response time  
- ❤️ Patient survival probability  

This system simulates **real-world ambulance routing and smart healthcare logistics**, going beyond traditional shortest-path navigation.

---

## 🌍 Project Vision

In real emergencies, the nearest hospital is not always the safest option.

HeartSafe Route AI models this complexity by forcing AI agents to balance:

- ⏱️ Fastest arrival time  
- 🫁 Lowest pollution exposure  
- 🚦 Traffic-aware routing  
- 🏥 Hospital readiness & capacity  
- ❤️ Patient survival safety score  

👉 The goal is to build a **realistic healthcare decision-making RL benchmark environment**.

---

## 🧠 Why This Project Matters

Traditional routing systems fail in critical medical emergencies because they ignore real-world constraints.

This project introduces AI-driven decision making where:

- A nearby hospital may be full  
- A fast route may pass through polluted zones  
- A low-traffic route may still be medically unsafe  
- AI must optimize survival probability, not just distance  

👉 This makes it a **high-impact reinforcement learning benchmark for healthcare AI systems**.

---

## ⚙️ Tech Stack

### 🧠 Backend (OpenEnv RL Engine)
- Python 3.10+
- FastAPI
- NumPy
- Scikit-learn (simulation + scoring logic)

### 🎨 Frontend (Hackathon Demo UI)
- React + Vite
- TailwindCSS
- Leaflet.js (interactive maps)
- Chart.js (AI analytics visualization)

### ☁️ Infrastructure
- Docker (containerized execution)
- Hugging Face Spaces (deployment target)
- OpenEnv specification compliant architecture

---

## 🏗️ System Modules

- 🚑 **Routing Agent** → RL-based decision engine  
- 🏥 **Hospital Service** → Capacity + emergency readiness simulation  
- 🚦 **Traffic Service** → Dynamic congestion modeling  
- 🫁 **Pollution Service** → Air quality impact zones  
- 🌍 **Simulation Engine** → Environment state generator  
- 📊 **Reward System** → Survival-based scoring function  

---

## 🎯 Key Features

- 🧠 Reinforcement Learning-based route optimization  
- 🚑 Emergency ambulance simulation environment  
- 🏥 Dynamic hospital capacity modeling  
- 🚦 Real-time traffic simulation  
- 🫁 Pollution-aware navigation  
- ❤️ Survival probability reward system  
- 📊 AI decision analytics dashboard  

---

## 📦 Project Structure

```bash
HeartSafe-Route-AI/
│
├── backend/
│   ├── main.py                # FastAPI app (OpenEnv server)
│   ├── routing_agent.py       # AI routing policy + reward logic
│   ├── hospital_service.py    # Hospital simulation engine
│   ├── traffic_service.py     # Traffic generator
│   ├── pollution_service.py   # Pollution zone engine
│   └── simulation.py          # RL environment core
│
├── frontend/
│   ├── src/
│   │   ├── components/
│   │   ├── pages/
│   │   ├── services/
│   │   ├── App.jsx
│   │   └── main.jsx
│
├── openenv.yaml               # OpenEnv configuration
├── inference.py               # Baseline evaluation script
├── Dockerfile
└── README.md
```
# 🫀 OpenEnv Compliant RL Environment

An AI-powered emergency navigation system where reinforcement learning agents learn to choose optimal hospital routes under real-world constraints like traffic, pollution, and hospital capacity.

---

## 🧩 OpenEnv Compliance (Core Design)

This environment strictly follows the OpenEnv specification.

---

## 🔁 Required API

```python
reset() -> Observation
step(action) -> Observation, Reward, Done, Info
state() -> CurrentState
```
## 📦 Observation Space

Each observation includes:

{{
  "user_location": [lat, lng],
  "hospitals": [...],
  "traffic_map": [...],
  "pollution_zones": [...],
  "time_elapsed": float,
  "emergency_level": int
}

## 🎮 Action Space

Agent selects:

{
  "selected_route": "fastest | safest | balanced",
  "target_hospital_id": int,
  "routing_priority_weights": {
      "time": float,
      "traffic": float,
      "pollution": float,
      "hospital_capacity": float
  }
}

## 🏥 Core Simulation Features

🚑 Realistic Emergency Environment

Live hospital capacity simulation
Traffic congestion generator
Pollution exposure heatmaps
Dynamic route scoring engine

🧪 RL Tasks (3 Difficulty Levels)
🟢 Task 1 — Emergency Fast Route (Easy)
Objective: Reach nearest hospital quickly
Constraints: Basic traffic + distance
🟡 Task 2 — Safe Hospital Selection (Medium)
Objective: Avoid overcrowded hospitals
Includes: capacity + traffic + distance balancing
🔴 Task 3 — Survival Optimization (Hard)
Objective: Maximize patient survival probability

Includes:

pollution exposure penalty
traffic delay risk
hospital readiness score
time-critical constraints
🧮 Reward Function Design

Reward is continuous and multi-objective:

reward =
    (w1 * -travel_time)
  + (w2 * hospital_availability)
  + (w3 * -traffic_delay)
  + (w4 * -pollution_exposure)
  + (w5 * survival_probability)
Penalties:
High pollution → negative reward
Delays → exponential penalty
Full hospital → strong penalty
🤖 AI Routing Agent

The baseline agent simulates intelligent decision-making:

Evaluates all hospital options
Computes weighted risk score
Chooses optimal survival path
Learns trade-offs between speed vs safety
📊 Grading System (OpenEnv Agent Evaluator)

Each task returns a score:

0.0 → failed mission
0.5 → partial success
1.0 → optimal route selected
📡 API Endpoints (FastAPI)
🏥 GET /hospitals

Returns nearby hospitals:

{
  "name": "City Heart Hospital",
  "lat": 23.25,
  "lng": 77.41,
  "distance": 3.2,
  "cardiac_specialization": true,
  "capacity": 80
}
🚦 GET /traffic

Returns traffic intensity:

low
medium
high
🌫️ GET /pollution

Returns pollution zones:

safe
moderate
dangerous
🧠 POST /compute-route

Input:

{
  "user_lat": 23.2,
  "user_lng": 77.4
}

Output:

{
  "best_hospital": "...",
  "estimated_time": 12,
  "safety_score": 0.87,
  "routes": {
    "fastest": {...},
    "safest": {...},
    "balanced": {...}
  }
}
🧪 Baseline Evaluation (inference.py)

Runs standardized OpenEnv evaluation:

Required environment variables:
OPENAI_API_KEY=your_key
API_BASE_URL=your_endpoint
MODEL_NAME=gpt-4o-mini

Run:
python inference.py
Output format:
[START]
[STEP] Observation processed
[STEP] Action selected
[STEP] Reward received
[END] Score: 0.82
🐳 Docker Deployment
Build:
docker build -t heartsafe-ai .
Run:
docker run -p 7860:7860 heartsafe-ai
🚀 Run Locally
Backend
uvicorn backend.main:app --reload
Frontend
cd frontend
npm install
npm run dev
☁️ Hugging Face Deployment
Fully containerized OpenEnv Space
Auto-start FastAPI server
Validated with openenv validate
Compatible with 2 vCPU / 8GB RAM constraints
🧠 Key Innovations
Real-world emergency healthcare RL simulation
Multi-objective reward system (not toy environment)
Hospital capacity-aware routing
Pollution + traffic + survival modeling
OpenEnv-compliant structured evaluation system
Agent grading with deterministic scoring
🏁 Final Goal

This environment enables AI agents to learn:

👉 How to save lives under real-world emergency constraints
