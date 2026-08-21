"""Durable MongoDB-backed worker for private persona analysis jobs.

Run in a worker-capable environment with:
    python -m Backend.PersonaWorker
"""

from __future__ import annotations

import argparse
import logging
import os
import socket
import time
import uuid

from Backend.MongoStore import StoreUnavailable
from Backend.Persona import process_persona_run


logger = logging.getLogger(__name__)


def worker_id() -> str:
    return f"{socket.gethostname()}-{os.getpid()}-{uuid.uuid4().hex[:8]}"


def run_worker(once: bool = False, poll_seconds: float = 3.0) -> int:
    identity = worker_id()
    processed = 0
    logger.info("Persona worker %s started", identity)
    while True:
        try:
            claimed = process_persona_run(worker_id=identity)
            if claimed:
                processed += 1
                if once:
                    return processed
                continue
        except StoreUnavailable as exc:
            logger.warning("Persona worker is waiting for MongoDB: %s", exc)
        except Exception:
            logger.exception("Persona worker loop failed; polling will continue.")
        if once:
            return processed
        time.sleep(max(0.5, poll_seconds))


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the Nexa persona background worker.")
    parser.add_argument("--once", action="store_true", help="Claim at most one available job, then exit.")
    parser.add_argument("--poll-seconds", type=float, default=3.0)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    run_worker(once=args.once, poll_seconds=args.poll_seconds)


if __name__ == "__main__":
    main()
