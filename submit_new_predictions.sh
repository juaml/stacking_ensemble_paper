#!/bin/bash
# predict new data

CPUS=1
test_path="/data/project/avg_vs_pred_stack/data"

pwd=`pwd`
logdir=$pwd/log
mkdir -p $logdir

declare -a datasets=(1000Gehirne CamCAN eNKI IXI)
show() {
    for test_data in "${datasets[@]}"; do
        
        local -a results=()
        let idx=$2
        for (( j = 0; j < $1; j++ )); do
            if (( idx % 2 )); then results=("${results[@]}" "${datasets[$j]}"); fi
            let idx\>\>=1
        done

        train_data=$(IFS="," ; echo "${results[*]}")
        if [[ "$train_data" != *"$test_data"* && ${train_data} == *","* ]]; then
            # echo \'$train_data\' $test_data
            printf "universe = vanilla\n"
            printf "request_cpus = ${CPUS}\n"
            printf "request_memory = 10G\n"
            printf "request_disk = 1G\n"
            printf "initialdir = ${pwd}\n"
            printf "executable = /data/project/avg_vs_pred_stack/util/R-4.1.0/bin/Rscript\n"
            printf "transfer_executable = false\n"
            printf "arguments = \"--max-ppsize=500000 predict_newsite.R \'$train_data\' $test_path $test_data\"\n"
            printf "output = ${logdir}/predict.${train_data}_${test_data}.\$(Cluster).\$(Process).out\n"
            printf "error = ${logdir}/predict.${train_data}_${test_data}.\$(Cluster).\$(Process).err\n"
            printf "log = ${logdir}/predict.${train_data}_${test_data}.\$(Cluster).\$(Process).log\n"
            printf "queue\n"
            printf "###\n\n"
        fi
        train_data=$(IFS=- ; echo "${results[*]}")
        if [[ "$train_data" != *"$test_data"* ]]; then
            # echo \'$train_data\' $test_data
            printf "universe = vanilla\n"
            printf "request_cpus = ${CPUS}\n"
            printf "request_memory = 10G\n"
            printf "request_disk = 1G\n"
            printf "initialdir = ${pwd}\n"
            printf "executable = /data/project/avg_vs_pred_stack/util/R-4.1.0/bin/Rscript\n"
            printf "transfer_executable = false\n"
            printf "arguments = \"--max-ppsize=500000 predict_newsite.R $train_data $test_path $test_data\"\n"
            printf "output = ${logdir}/predict.${train_data}_${test_data}.\$(Cluster).\$(Process).out\n"
            printf "error = ${logdir}/predict.${train_data}_${test_data}.\$(Cluster).\$(Process).err\n"
            printf "log = ${logdir}/predict.${train_data}_${test_data}.\$(Cluster).\$(Process).log\n"
            printf "queue\n"
            printf "###\n\n"
        fi
    done
}

let n=${#datasets[@]}
for (( i = 1; i < 2**n; i++ )); do
    show $n $i
done