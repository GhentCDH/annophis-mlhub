from fastapi import FastAPI
from scalar_fastapi import get_scalar_api_reference

CUSTOM_CSS = """
[class*="powered-by"] { display: none !important; }
.agent-button-container {
  display: none !important;
}
a.no-underline,
.scalar-mcp-layer,
button.bg-sidebar-b-search:nth-child(3),
.open-api-client-button
{
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
