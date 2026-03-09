from fastapi import FastAPI
from scalar_fastapi import get_scalar_api_reference

CUSTOM_CSS = """
:root {
  --scalar-color-1: #1a1a2e;
  --scalar-color-accent: #e94560;
  --scalar-background-1: #fafafa;
  --scalar-border-color: #e0e0e0;
}
.dark-mode {
  --scalar-background-1: #16213e;
  --scalar-background-2: #1a1a2e;
  --scalar-color-1: #e0e0e0;
}
[class*="powered-by"] { display: none !important; }
.agent-button-container {
  display: none !important;
}
"""


def add_scalar_docs(app: FastAPI) -> None:
    @app.get("/docs", include_in_schema=False)
    async def scalar_docs():
        return get_scalar_api_reference(
            openapi_url=app.openapi_url,
            title=app.title,
            dark_mode=True,
            custom_css=CUSTOM_CSS,
        )
