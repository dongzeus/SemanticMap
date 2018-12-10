#!/bin/bash
OMP_NUM_THREADS=10
export OMP_NUM_THREADS
#

DIRNAME=`date '+%Y-%m-%d-%H-%M-%S'`
LOGPATH="./log/$DIRNAME"
LOGFILE="$LOGPATH/log.txt"
if [ ! -d "$LOGPATH" ];then
mkdir "$LOGPATH"
fi


 python -u ./main.py\
                    --epoch 100\
                    --batchSize 1000\
                    --lr 1e-4\
                    --batchModelSave 5000\
                    --logPath "$LOGPATH"\
                    --checkPoint ""
