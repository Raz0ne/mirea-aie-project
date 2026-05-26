"""`python -m src.service` — run the FastAPI service via uvicorn."""
from __future__ import annotations

import uvicorn

from src.utils.config import ServiceSettings


def main():
    settings = ServiceSettings()
    uvicorn.run(
        "src.service.app:app",
        host=settings.app_host,
        port=settings.app_port,
        log_level=settings.log_level.lower(),
        reload=False,
    )


if __name__ == "__main__":
    main()
