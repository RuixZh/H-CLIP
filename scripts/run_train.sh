#!/usr/bin/env bash
if [ -f $1 ]; then
  config=$1
else
  echo "need a config file"
  exit
fi

now=$(date +"%Y%m%d_%H%M%S")
python3 -u train.py  --config ${config} --log_time $now 2>&1
# --mail-user=mengmengwang@zju.edu.cn --mail-type=ALL -x node86
