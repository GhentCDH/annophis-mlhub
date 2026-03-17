import logging

import uvicorn
from rich.logging import RichHandler

from konekaare.config import settings

logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(rich_tracebacks=True)],
)


def main():
    uvicorn.run(
        "konekaare.app:app",
        host=settings.host,
        port=settings.port,
        reload=settings.debug,
    )


if __name__ == "__main__":
    main()
