#!/usr/bin/env bash

mkdir data

# Download GloVe embeddings
EMBDIR="data/embeddings"
mkdir -p $EMBDIR
wget -P $EMBDIR https://allennlp.s3.amazonaws.com/datasets/glove/glove.840B.300d.txt.gz

# Download LIF dataset
DATA_DIR="data"
# Link:  https://drive.google.com/file/d/1LN2HQRLRuDt8zw8TM8RqP9wyYfY6Swun/view?usp=sharing
cd $DATA_DIR
wget --load-cookies /tmp/cookies.txt "https://drive.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://drive.google.com/uc?export=download&id=1LN2HQRLRuDt8zw8TM8RqP9wyYfY6Swun' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1LN2HQRLRuDt8zw8TM8RqP9wyYfY6Swun" -O lif_v1.zip && rm -rf /tmp/cookies.txt
unzip lif_v1.zip
rm lif_v1.zip
cd ..


