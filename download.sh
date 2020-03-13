#!/usr/bin/env bash
BERT_DIR=$(pwd)/coinform_content_analysis/data/fine_tuned_models
echo "Create a folder BERT_DIR"
mkdir ${BERT_DIR}

## DOWNLOAD BERT
cd ${BERT_DIR}
wget https://storage.googleapis.com/bert_models/2018_10_18/uncased_L-12_H-768_A-12.zip -O "uncased_bert_base.zip"
unzip uncased_bert_base.zip
mv uncased_L-12_H-768_A-12/vocab.txt "${BERT_DIR}/"
rm *.zip
rm -rf uncased_L-12_H-768_A-12
