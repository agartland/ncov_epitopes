#!/bin/sh

IEDB_PREDICT="/home/agartlan/gitrepo/HLAPredCache/iedb_predict.py"
DATA_PATH="/fh/fast/gilbert_p/fg_data/ncov_epitopes/data"

#               --verbose \

for k in 8 9 10 11
do
$IEDB_PREDICT --method netmhcpan \
              --pep $DATA_PATH/ncov.$k.mers \
              --hla $DATA_PATH/ncov.hla \
              --out $DATA_PATH/ncov.$k.out \
              --cpus 12
done