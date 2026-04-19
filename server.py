"""W&B Automation webhook server.

Receives POST callbacks from W&B Automations and can trigger new training
runs, chain experiments, or perform any other action.

Start the server
----------------
    python server.py                          # default port 8000
    python server.py --port 9000
    python server.py --host 0.0.0.0 --port 8000   # expose to LAN

Make it reachable from W&B
--------------------------
W&B Automations require a publicly reachable HTTPS URL.  For local dev use
ngrok (or any tunnel):

    ngrok http 8000
    # copy the https://... forwarding URL

Then in W&B UI:
    Project → Automations → New Automation
      Trigger: e.g. "Run metric threshold" (mean_reward > 200)
      Action:  Webhook → <your ngrok URL>/webhook

Available endpoints
-------------------
POST /launch     Immediately queue a new training run with an optional config
                 body.  Accepts JSON {"config": {...}, "note": "..."}.

POST /webhook    Receive a W&B Automation event. The handler inspects the
                 event type and dispatches:
                   - run.finished  → optional auto-relaunch
                   - run.failed    → log + optional retry
                   - (extend as needed)

GET  /status     Return the list of running/finished job subprocesses.

Security: set WEBHOOK_SECRET env-var to validate W&B's HMAC-SHA256 signature.
"""

import argparse
import hashlib
import hmac
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import uvicorn
from fastapi import BackgroundTasks, FastAPI, Header, HTTPException, Request
from fastapi.responses import JSONResponse

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

WEBHOOK_SECRET: str = os.environ.get("WEBHOOK_SECRET", "")  # set to validate HMAC
PYTHON_EXE: str = str(Path(sys.executable))
MAIN_PY: str = str(Path(__file__).parent / "main.py")

# In-memory job registry (pid → info dict).  A real deployment would use a DB.
_jobs: dict[int, dict[str, Any]] = {}

app = FastAPI(title="TORCS RL webhook server", version="0.1.0")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _verify_signature(body: bytes, signature_header: str | None) -> None:
    """Raise 401 if the HMAC-SHA256 signature from W&B does not match."""
    if not WEBHOOK_SECRET:
        return  # validation disabled — set WEBHOOK_SECRET to enable
    if not signature_header:
        raise HTTPException(status_code=401, detail="Missing X-Wandb-Signature header")
    expected = "sha256=" + hmac.new(
        WEBHOOK_SECRET.encode(), body, hashlib.sha256
    ).hexdigest()
    if not hmac.compare_digest(expected, signature_header):
        raise HTTPException(status_code=401, detail="Invalid signature")


def _launch_job(config: dict[str, Any], note: str = "") -> dict[str, Any]:
    """Spawn a training subprocess and return job info."""
    cmd = [PYTHON_EXE, MAIN_PY, "--launch"]

    # Pass scalar config values as CLI flags so apply_wandb_overrides picks them up.
    for key, value in config.items():
        if value is None:
            continue
        flag = f"--{key.replace('_', '-')}"
        if isinstance(value, bool):
            if value:
                cmd.append(flag)
        else:
            cmd.extend([flag, str(value)])

    proc = subprocess.Popen(cmd, cwd=str(Path(MAIN_PY).parent))
    info = {
        "pid": proc.pid,
        "cmd": cmd,
        "config": config,
        "note": note,
        "started_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "status": "running",
    }
    _jobs[proc.pid] = info
    print(f"[server] Launched job pid={proc.pid}  cmd={cmd}")
    return info


def _poll_jobs() -> None:
    """Update status of all tracked subprocesses (non-blocking)."""
    for info in _jobs.values():
        if info["status"] != "running":
            continue
        # We don't keep the Popen handle, so use os.waitpid with WNOHANG.
        try:
            pid, code = os.waitpid(info["pid"], os.WNOHANG)
            if pid != 0:
                info["status"] = "finished" if code == 0 else f"failed(rc={code >> 8})"
        except ChildProcessError:
            info["status"] = "finished"


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/status")
def status():
    """Return all tracked jobs."""
    _poll_jobs()
    return JSONResponse({"jobs": list(_jobs.values())})


@app.post("/launch")
async def launch(request: Request, background_tasks: BackgroundTasks):
    """Immediately launch a training run.

    Body (optional JSON):
        {
            "config": {"algo": "sac", "timesteps": 50000, "target_speed": 80},
            "note": "triggered manually"
        }
    """
    body = await request.body()
    payload: dict = {}
    if body:
        try:
            payload = json.loads(body)
        except json.JSONDecodeError:
            raise HTTPException(status_code=400, detail="Invalid JSON body")

    config = payload.get("config", {})
    note = payload.get("note", "manual launch")
    info = _launch_job(config, note=note)
    return JSONResponse({"launched": True, "job": info})


@app.post("/webhook")
async def webhook(
    request: Request,
    background_tasks: BackgroundTasks,
    x_wandb_signature: str | None = Header(default=None),
):
    """Receive a W&B Automation event.

    W&B posts JSON with at minimum:
        {
            "event_type": "run.finished" | "run.failed" | ...,
            "event_author": "<username>",
            "project": {"name": "...", "entity": "..."},
            "run": {"name": "...", "id": "...", "config": {...}, "summary": {...}}
        }

    Add cases below for any event type you want to handle.
    """
    body = await request.body()
    _verify_signature(body, x_wandb_signature)

    try:
        event = json.loads(body)
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Invalid JSON")

    event_type: str = event.get("event_type", "")
    run_info: dict = event.get("run", {})
    run_name: str = run_info.get("name", "unknown")
    run_summary: dict = run_info.get("summary", {})
    run_config: dict = run_info.get("config", {})

    print(f"[server] Webhook received: event_type={event_type!r} run={run_name!r}")

    # -----------------------------------------------------------------------
    # Dispatch on event type — extend this section to add your own logic.
    # -----------------------------------------------------------------------
    if event_type == "run.finished":
        mean_reward = run_summary.get("eval/mean_reward")
        print(f"[server] Run {run_name!r} finished. mean_reward={mean_reward}")

        # Example: auto-relaunch with higher timesteps if reward looks promising.
        if mean_reward is not None and mean_reward > 50:
            new_config = dict(run_config)
            new_config["timesteps"] = int(run_config.get("timesteps", 6000)) * 2
            new_config["run_name"] = f"{run_name}-followup"
            background_tasks.add_task(_launch_job, new_config, note="auto-followup")
            return JSONResponse({"action": "followup_launched", "config": new_config})

        return JSONResponse({"action": "none", "reason": "reward below threshold"})

    if event_type == "run.failed":
        print(f"[server] Run {run_name!r} failed. Retrying with same config.")
        retry_config = dict(run_config)
        retry_config["run_name"] = f"{run_name}-retry"
        background_tasks.add_task(_launch_job, retry_config, note="auto-retry")
        return JSONResponse({"action": "retry_launched"})

    # Unhandled event types — acknowledge receipt so W&B doesn't retry.
    return JSONResponse({"action": "ignored", "event_type": event_type})


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def _parse_args():
    parser = argparse.ArgumentParser(description="W&B Automation webhook server")
    parser.add_argument("--host", default="127.0.0.1", help="Bind host (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8000, help="Bind port (default: 8000)")
    parser.add_argument("--reload", action="store_true", help="Enable uvicorn auto-reload (dev mode)")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    print(f"[server] Starting on http://{args.host}:{args.port}")
    print(f"[server] Webhook secret {'set' if WEBHOOK_SECRET else 'NOT SET (set WEBHOOK_SECRET env-var)'}.")
    uvicorn.run("server:app", host=args.host, port=args.port, reload=args.reload)
