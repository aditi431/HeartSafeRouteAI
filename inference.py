"""
Inference Script — EcoNav AI Exposure Credit Environment
========================================================

MANDATORY:
1. Environment variables: API_BASE_URL, MODEL_NAME, HF_TOKEN
2. Uses OpenAI Client for all LLM calls
3. Structured logging: [START]/[STEP]/[END]

Runs all 3 tasks and produces reproducible scores.
"""

import os
import sys
import time
import traceback

import requests
from openai import OpenAI

# ---------------------------------------------------------------------------
# Environment config
# ---------------------------------------------------------------------------

ENV_URL = os.getenv("ENV_URL", "http://localhost:7860")
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "meta-llama/Llama-3.1-8B-Instruct")
HF_TOKEN = os.getenv("HF_TOKEN")


# ---------------------------------------------------------------------------
# Environment interaction helpers
# ---------------------------------------------------------------------------


def env_reset(task_id: str = "easy_route") -> dict:
    """Reset environment with retry logic."""
    max_retries = 5
    for i in range(max_retries):
        try:
            resp = requests.post(
                f"{ENV_URL}/reset",
                json={"task_id": task_id},
                timeout=30,
            )
            resp.raise_for_status()
            return resp.json()
        except Exception as e:
            if i == max_retries - 1:
                raise
            print(f"  Waiting for environment... (attempt {i+1}/{max_retries}): {e}")
            time.sleep(3)
    return {}


def env_step(city: str) -> dict:
    resp = requests.post(
        f"{ENV_URL}/step",
        json={"action": {"city": city}},
        timeout=10,
    )
    resp.raise_for_status()
    return resp.json()


def env_grade() -> dict:
    resp = requests.get(f"{ENV_URL}/grade", timeout=10)
    resp.raise_for_status()
    return resp.json()


def env_tasks() -> list:
    """Fetch tasks with retry logic."""
    max_retries = 8
    for i in range(max_retries):
        try:
            resp = requests.get(f"{ENV_URL}/tasks", timeout=15)
            resp.raise_for_status()
            return resp.json()["tasks"]
        except Exception as e:
            if i == max_retries - 1:
                raise
            print(f"Waiting for environment... (attempt {i+1}/{max_retries}): {e}")
            time.sleep(4)
    return []


# ---------------------------------------------------------------------------
# LLM Agent
# ---------------------------------------------------------------------------


def get_llm_action(observation: dict) -> str:
    """Ask the LLM to choose the next city based on observation."""
    neighbors = observation.get("neighbors", [])
    if not neighbors:
        return "F"

    neighbor_text = "\n".join(
        f"  - {n['city']} ({n['city_name']}): AQI={n['aqi']}, "
        f"Grade={n['grade']}, Credits={n['credit_delta']:+d}, "
        f"Distance={n['distance']}km"
        for n in neighbors
    )

    prompt = f"""You are an AI navigation agent in the EcoNav environment.
Your goal: reach {observation["destination"]} ({observation["destination_name"]}) while maximizing exposure credits.

Current state:
- Location: {observation["current_city"]} ({observation["current_city_name"]})
- Destination: {observation["destination"]} ({observation["destination_name"]})
- Credits: {observation["exposure_credits"]}
- Steps: {observation["steps_taken"]}/{observation["max_steps"]}
- Visited: {observation["visited"]}
- Total exposure: {observation["total_exposure"]}

Available moves:
{neighbor_text}

Rules:
- Choose cities with lower AQI (Grade A/B) to EARN credits
- Avoid high-AQI cities (Grade D/E/F) that LOSE credits
- You MUST reach the destination before running out of steps
- Balance: short path vs clean-air path

Reply with ONLY the city code (single letter like "E") — nothing else."""

    valid = [n["city"] for n in neighbors]

    try:
        client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=10,
            temperature=0.0,
        )
        raw = response.choices[0].message.content or ""
        action = raw.strip().upper().split()[0] if raw.strip() else ""
        # Clean up
        action = action.replace('"', "").replace(".", "").replace(",", "").replace("'", "")

        if action in valid:
            return action
    except Exception as exc:
        print(f"  LLM call failed ({exc}). Using heuristic fallback.")

    # Heuristic fallback: prefer destination if reachable, else lowest AQI unvisited
    visited = set(observation.get("visited", []))
    dest = observation.get("destination", "")

    if dest in valid:
        return dest

    # Pick unvisited neighbor with lowest AQI
    unvisited = [n for n in neighbors if n["city"] not in visited]
    candidates = unvisited if unvisited else neighbors

    # Sort by: destination proximity (credit_delta desc), then AQI
    best = sorted(candidates, key=lambda n: n["aqi"])
    return best[0]["city"] if best else valid[0]


# ---------------------------------------------------------------------------
# Main Evaluation Loop
# ---------------------------------------------------------------------------


def run_evaluation():
    try:
        tasks = env_tasks()
    except Exception as e:
        print(f"Error fetching tasks: {e}")
        traceback.print_exc()
        return

    for task in tasks:
        task_id = task["id"]
        try:
            print(f"[START] task_id={task_id}")

            obs = env_reset(task_id)
            done = obs.get("done", False)
            step_count = 0

            while not done:
                action = get_llm_action(obs)

                try:
                    result = env_step(action)
                except Exception as step_err:
                    print(f"  Step error: {step_err}")
                    # Try first valid neighbor as fallback
                    neighbors = obs.get("neighbors", [])
                    if neighbors:
                        fallback = neighbors[0]["city"]
                        result = env_step(fallback)
                    else:
                        break

                obs = result["observation"]
                reward = result["reward"]
                done = result["done"]

                print(f"[STEP] step={step_count} action={action} reward={reward:.4f}")
                step_count += 1

                if step_count > 25:
                    break

            # Grade episode
            try:
                grade_result = env_grade()
                score = grade_result.get("score", 0.0001)
                grade_letter = grade_result.get("grade_letter", "F")
                reached = grade_result.get("reached_destination", False)
            except Exception:
                score = 0.0001
                grade_letter = "F"
                reached = False

            print(f"[END] score={score:.4f} grade={grade_letter} reached={reached}")

        except Exception as e:
            print(f"Error during task {task_id}: {e}")
            traceback.print_exc()
            print(f"[END] score=0.0001 grade=F reached=False")


if __name__ == "__main__":
    if not HF_TOKEN:
        print("Warning: HF_TOKEN not set. LLM calls will use heuristic fallback.")
    run_evaluation()
