#!/usr/bin/env bash

# "$0" strip file name:  ../dir/test.sh -> ../dir/
# "$(dirname "$0")" get directory name:  ../dir/test.sh -> ../dir/
# pwd -P print absolute path of the working directory:  ->  /homeuser/dir/
# SCRIPTPATH="$( cd "$(dirname "$0")" ; pwd -P )" -> /home/user/dir/
SCRIPTPATH="$( cd "$(dirname "$0")" ; pwd -P )"
./build.sh

# create volumn name
VOLUME_SUFFIX=$(dd if=/dev/urandom bs=32 count=1 | md5sum | cut --delimiter=' ' --fields=1)
MEM_LIMIT="30g"  # Maximum is currently 30g, configurable in your algorithm image settings on grand challenge

docker volume create synthrad_algorithm-output-$VOLUME_SUFFIX

# run algorithm
# Do not change any of the parameters to docker run, these are fixed
docker run --rm \
        --memory="${MEM_LIMIT}" \
        --memory-swap="${MEM_LIMIT}" \
        --network="none" \
        --cap-drop="ALL" \
        --security-opt="no-new-privileges" \
        --shm-size="128m" \
        --pids-limit="256" \
        -v $SCRIPTPATH/test:/input \
        -v synthrad_algorithm-output-$VOLUME_SUFFIX:/output/ \
        synthrad_algorithm

# display results
docker run --rm \
        -v synthrad_algorithm-output-$VOLUME_SUFFIX:/output/ \
        python:3.12-slim cat /output/results.json | python3 -m json.tool

# validate outputs
docker run --rm \
        -v $SCRIPTPATH/test:/input \
        -v synthrad_algorithm-output-$VOLUME_SUFFIX:/output/ \
        python:3.12-slim python -c "import json, sys; f1 = json.load(open('/output/results.json')); f2 = json.load(open('/input/expected_output.json')); sys.exit(f1 != f2);"

# report outputs
if [ $? -eq 0 ]; then
    echo "Tests successfully passed..."
else
    echo "Expected output was not found..."
fi
# test/
# ├── images/
# │   ├── mri/          # Input MRI images
# │   └── body/         # Body mask images
# ├── region.json       # Region information
# └── expected_output.json  # Expected results

# docker volume rm synthrad_algorithm-output-$VOLUME_SUFFIX
