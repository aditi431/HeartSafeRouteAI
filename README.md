# EcoNav AI — Exposure Credit Navigator

> **OpenEnv-compliant RL Environment** for training AI agents to navigate Indian cities while minimising pollution exposure and maximising Exposure Credits.

## Overview

EcoNav AI simulates a real-world task: **pollution-aware route planning across Indian cities**. An AI agent navigates a graph of 9 Indian cities (Delhi, Jaipur, Agra, Varanasi, Lucknow, Kolkata, Chandigarh, Bhopal, Patna) making sequential routing decisions. Each city has a realistic AQI (Air Quality Index) profile, and the agent earns or loses **Exposure Credits** based on the air quality of its chosen path.

This environment addresses a genuine public health challenge — air pollution kills over 2 million people annually in India. Training agents to find clean-air routes has immediate real-world value.

## Key Features

- **Real-world domain**: Pollution-aware navigation using realistic Indian city AQI data
- **OpenEnv spec compliant**: Full `step()` / `reset()` / `state()` / `grade()` API
- **3 tasks with difficulty progression**: Easy → Medium → Hard
- **Rich reward shaping**: Credit-based rewards, distance penalties, destination bonuses
- **Deterministic grading**: Reproducible scores 0.0-1.0
- **Interactive frontend**: Live graph visualization, manual play, and auto-agent mode
- **Self-contained**: No external API calls required (embedded AQI profiles)

## Action & Observation Spaces

### Observation Space

| Field | Type | Description |
|-------|------|-------------|
| `current_city` | string | Current city code (A-I) |
| `destination` | string | Target city code |
| `visited` | list[str] | Cities already visited |
| `neighbors` | list[dict] | Available moves with AQI, grade, credits, distance |
| `exposure_credits` | int | Current credit balance (starts at 100) |
| `total_exposure` | float | Cumulative pollution exposure |
| `steps_taken` | int | Steps used so far |
| `max_steps` | int | Maximum steps allowed |
| `done` | bool | Episode finished? |

### Action Space

| Field | Type | Description |
|-------|------|-------------|
| `city` | string | City code to move to (must be a valid neighbor) |

### Reward Function

```
reward = credit_delta (AQI-grade based: +10 to -50)
       + distance_penalty (-distance/1000)
       + destination_bonus (+100 if reached)
       + revisit_penalty (-10 if city already visited)
       + timeout_penalty (-50 if steps exhausted without reaching destination)
```

### Grade System

| AQI Range | Grade | Credits/Segment |
|-----------|-------|-----------------|
| ≤ 50      | A     | +10             |
| 51-100    | B     | +5              |
| 101-150   | C     | -5              |
| 151-200   | D     | -15             |
| 201-300   | E     | -30             |
| 300+      | F     | -50             |

## Tasks

### Task 1: Easy — Delhi to Kolkata (15 steps)
Navigate from Delhi (A) to Kolkata (F) with a generous step budget. Multiple paths available. Passing score: 0.5.

### Task 2: Medium — Delhi to Kolkata (8 steps)
Same origin/destination but with a tight step budget. Requires balancing speed vs. clean-air routes. Passing score: 0.6.

### Task 3: Hard — Agra to Kolkata (6 steps)
Start from heavily-polluted Agra (C), reach Kolkata (F) in just 6 steps while maximizing credits. Very challenging — requires optimal pollution-dodging strategy. Passing score: 0.7.

## Scoring Breakdown

Final score is 0.0-1.0, computed as:
- **Destination reached** (40%): Binary — did the agent reach the target?
- **Credit performance** (30%): Final credits relative to starting balance
- **Exposure minimization** (20%): Lower cumulative exposure = better
- **Step efficiency** (10%): Fewer steps = higher efficiency score

## Setup & Usage

### Prerequisites
- Python 3.10+
- Docker (optional, for containerized deployment)

### Local Development

```bash
# Clone and install
git clone <repo-url>
cd econav-exposure-credit
pip install -e .

# Run the server
python server/app.py
# Server starts at http://localhost:7860

# Open frontend
# Visit http://localhost:7860/app in your browser
```

### Docker

```bash
docker build -t econav-ai .
docker run -p 7860:7860 econav-ai
```

### Run Inference

```bash
export HF_TOKEN="your-token"
export API_BASE_URL="https://router.huggingface.co/v1"
export MODEL_NAME="meta-llama/Llama-3.1-8B-Instruct"
export ENV_URL="http://localhost:7860"

python inference.py
```

### API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/reset` | POST | Reset environment, start new episode |
| `/step` | POST | Execute action, get observation + reward |
| `/state` | GET | Get current episode state |
| `/grade` | GET | Grade completed episode (0.0-1.0) |
| `/tasks` | GET | List all available tasks |
| `/tasks/{id}` | GET | Get specific task details |
| `/api/v1/graph` | GET | Get city graph for visualization |
| `/docs` | GET | Interactive API documentation |

## Baseline Scores

| Task | Heuristic Agent | LLM Agent (Llama-3.1-8B) |
|------|-----------------|--------------------------|
| easy_route | ~0.72 | ~0.78 |
| medium_route | ~0.55 | ~0.65 |
| hard_pollution_dodge | ~0.42 | ~0.52 |

## City Network

```
           G (Chandigarh, AQI ~75)
          / 
    A (Delhi, AQI ~185) ——— B (Jaipur, AQI ~120)
     \                       \
      C (Agra, AQI ~170) ——— D (Varanasi, AQI ~155) ——— I (Patna, AQI ~160)
       \                    / \                            \
        E (Lucknow, ~140) /   F (Kolkata, AQI ~95)         F
         \               /
          H (Bhopal, ~90)
```

## License

MIT
