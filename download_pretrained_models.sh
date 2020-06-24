#!/usr/bin/env bash

# Download the pretrained Models

# 3-way Attentive Pooling Network
#
AP_MODEL_DIR="data/pretrained-model/3way_ap"
mkdir -p $AP_MODEL_DIR
cd $AP_MODEL_DIR
# Link:  https://drive.google.com/file/d/1oiJrQRxu7VSHrjPXGGb8iNhYAXJBjLkZ/view?usp=sharing
wget --load-cookies /tmp/cookies.txt "https://drive.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://drive.google.com/uc?export=download&id=1oiJrQRxu7VSHrjPXGGb8iNhYAXJBjLkZ' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1oiJrQRxu7VSHrjPXGGb8iNhYAXJBjLkZ" -O model.tar.gz && rm -rf /tmp/cookies.txt
cd ../../..


# BERT baseline model
#
B_MODEL_DIR="data/pretrained-model/bert-baseline"
mkdir -p $B_MODEL_DIR
cd $B_MODEL_DIR
# Link:  https://drive.google.com/file/d/18FZMlzV7fI7r4bQV5IfXWGRyn-xDPlj-/view?usp=sharing
wget --load-cookies /tmp/cookies.txt "https://drive.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://drive.google.com/uc?export=download&id=18FZMlzV7fI7r4bQV5IfXWGRyn-xDPlj-' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=18FZMlzV7fI7r4bQV5IfXWGRyn-xDPlj-" -O model.tar.gz && rm -rf /tmp/cookies.txt
cd ../../..
