import logging

import uvicorn

from konekaare.config import settings

logging.basicConfig(level=logging.INFO)


def main():
    uvicorn.run(
        "konekaare.app:app",
        host=settings.host,
        port=settings.port,
        reload=settings.debug,
    )


if __name__ == "__main__":
    main()
