🫀 HeartSafe Route AI
AI-Powered Emergency Navigation System for Heart Patients (OpenEnv RL Environment)

🚨 An OpenEnv-compliant Reinforcement Learning environment where AI agents learn to save lives by choosing the safest and fastest cardiac emergency routes under real-world constraints like traffic, pollution, and hospital capacity.

🌍 Project Overview

HeartSafe Route AI transforms emergency navigation into a real-world reinforcement learning problem.

Instead of simple shortest-path routing, agents must optimize:

⏱️ Fastest arrival time
🫁 Lowest pollution exposure
🚦 Traffic-aware routing
🏥 Hospital availability & cardiac readiness
❤️ Patient survival safety score

This environment simulates real emergency decision-making systems used in ambulance routing and smart healthcare logistics.

🧠 Why This Project Matters

In real emergencies, reaching the nearest hospital is NOT always the safest option.

HeartSafe Route AI models real-world complexity:

A nearby hospital may be full
A fast route may pass through high pollution zones
A low-traffic route may still be medically unsafe
AI must balance speed vs survival probability

👉 This makes it a high-value RL benchmark environment for healthcare AI systems

⚙️ Tech Stack
Backend (OpenEnv RL Engine)
Python 3.10+
FastAPI
NumPy
Scikit-learn (simulation + scoring logic)
Frontend (Hackathon Demo UI)
React + Vite
TailwindCSS
Leaflet.js (interactive maps)
Chart.js (AI analytics visualization)
Infra
Docker (containerized execution)
Hugging Face Spaces (deployment target)
OpenEnv specification compliance
📁 Project Structure
## 📁 Project Structure

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
🧩 OpenEnv Compliance (Core Design)

This environment strictly follows OpenEnv spec:

🔁 Required API
reset() -> Observation
step(action) -> Observation, Reward, Done, Info
state() -> CurrentState
📦 Observation Space

Each observation includes:

{
  "user_location": [lat, lng],
  "hospitals": [...],
  "traffic_map": [...],
  "pollution_zones": [...],
  "time_elapsed": float,
  "emergency_level": int
}
🎮 Action Space

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
🏥 Core Simulation Features
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
