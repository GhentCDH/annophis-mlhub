export ANNOHUB_INLINE_CONTEXT:="true"
export LOG_LEVEL:="DEBUG"

image_name:="annophis_mlhub"

start $ANNOHUB_DEBUG="true":
  uv run main.py

# build the docker image & export to into a gzipped file
build:
  #!/usr/bin/env bash

  docker build -t {{image_name}}:latest .
  docker save {{image_name}}:latest | gzip > {{image_name}}.tar.gz

# push the built image to dockqas1
push server:
  scp {{image_name}}.tar.gz {{server}}:{{image_name}}.tar.gz

  @echo "do this on {{server}}:"
  @echo -e "\t\x1b[1mdocker load < {{image_name}}.tar.gz\x1b[0m"
