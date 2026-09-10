docker run \
  --rm \
  -it \
  --gpus all \
  --shm-size=2g \
  --mount "type=bind,source=$PWD,target=/workspace" \
  --workdir /workspace \
  --name idcard \
  ultralytics/ultralytics:latest \
  bash