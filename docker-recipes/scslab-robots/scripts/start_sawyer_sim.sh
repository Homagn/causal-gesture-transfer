#!/bin/bash
sudo nvidia-docker run --rm -ti --mount type=bind,source=/home/homagni/sawyer/,target=/sawyer --net=host --ipc=host -e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix --env="QT_X11_NO_MITSHM=1" scslab-robots /root/sawyer_entrypoint.sh