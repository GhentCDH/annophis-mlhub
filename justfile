export ANNOHUB_INLINE_CONTEXT:="true"
export LOG_LEVEL:="DEBUG"

start $ANNOHUB_DEBUG="true":
     uv run main.py
