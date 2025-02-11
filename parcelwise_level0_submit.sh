#!/bin/bash
# creates level0 models
# also generates data for level1 models
# if an argument is passed then its used as seed

if [[ -z $1 ]]; then
    seed="1"
else
    seed="$1"
fi

ddir="/data/project/avg_vs_pred_stack/data"
odir="/data/project/avg_vs_pred_stack/models"
method="glmnet"

pwd=`pwd`
logdir=$pwd/log
mkdir -p $logdir

declare -a datasets=(1000Gehirne CamCAN eNKI IXI)

show() {
    local -a results=()
    let idx=$2
    for (( j = 0; j < $1; j++ )); do
        if (( idx % 2 )); then results=("${results[@]}" "${datasets[$j]}"); fi
        let idx\>\>=1
    done
    data=$(IFS=- ; echo "${results[*]}")

  for fl in ${ddir}/*.csv; do
    parcel=`basename $fl .csv`
    printf "universe = vanilla\n"
    printf "request_cpus = 1\n"
    printf "request_memory = 10G\n"
    printf "request_disk = 1G\n"
    printf "initialdir = ${pwd}\n"
    printf "executable = /data/project/avg_vs_pred_stack/util/R-4.1.0/bin/Rscript\n"
    printf "transfer_executable = false\n"
    printf "arguments = \"--max-ppsize=500000 parcelwise_level0.R $fl $odir $method $data $seed\"\n"
    printf "output = ${logdir}/${parcel}_l0.${data}.\$(Cluster).\$(Process).out\n"
    printf "error = ${logdir}/${parcel}_l0.${data}.\$(Cluster).\$(Process).err\n"
    printf "log = ${logdir}/${parcel}_l0.${data}.\$(Cluster).\$(Process).log\n"
    printf "queue\n"
    printf "###\n\n"
  done
}

let n=${#datasets[@]}
for (( i = 1; i < 2**n; i++ )); do
    show $n $i
done


