#!/usr/bin/env bash

SCRIPTPATH="$( cd "$(dirname "$0")" ; pwd -P )"

# Only build the base image once (without copying the changing files)
if [[ "$(docker images -q synthrad_algorithm_base 2> /dev/null)" == "" ]]; then
    echo "Building base image (this only happens once)..."
    docker build -t synthrad_algorithm_base -f Dockerfile.base "$SCRIPTPATH"
fi

VOLUME_SUFFIX=$(dd if=/dev/urandom bs=32 count=1 | md5sum | cut --delimiter=' ' --fields=1)
MEM_LIMIT="30g"

docker volume create synthrad_algorithm-output-$VOLUME_SUFFIX

# Run with volume mounts for development (no rebuild needed)
docker run --rm \
        --gpus=all \
        --memory="${MEM_LIMIT}" \
        --memory-swap="${MEM_LIMIT}" \
        --network="none" \
        --cap-drop="ALL" \
        --security-opt="no-new-privileges" \
        --shm-size="128m" \
        --pids-limit="256" \
        -v $SCRIPTPATH/test:/input \
        -v $SCRIPTPATH/process.py:/opt/algorithm/process.py \
        -v $SCRIPTPATH/base_algorithm.py:/opt/algorithm/base_algorithm.py \
        -v $SCRIPTPATH/utils:/opt/algorithm/utils \
        -v $SCRIPTPATH/exp_configs:/opt/algorithm/exp_configs \
        -v $SCRIPTPATH/some_checkpoints:/opt/algorithm/some_checkpoints \
        -v synthrad_algorithm-output-$VOLUME_SUFFIX:/output/ \
        synthrad_algorithm_base

docker run --rm \
        -v synthrad_algorithm-output-$VOLUME_SUFFIX:/output/ \
        python:3.12-slim cat /output/results.json | python3 -m json.tool

docker run --rm \
        -v $SCRIPTPATH/test:/input \
        -v synthrad_algorithm-output-$VOLUME_SUFFIX:/output/ \
        python:3.12-slim python -c "import json, sys; f1 = json.load(open('/output/results.json')); f2 = json.load(open('/input/expected_output.json')); sys.exit(f1 != f2);"

if [ $? -eq 0 ]; then
    echo "Tests successfully passed..."
else
    echo "Expected output was not found..."
fi

# docker volume rm synthrad_algorithm-output-$VOLUME_SUFFIX 