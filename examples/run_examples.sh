#!/bin/bash

examples_dir=${1}
if [ -z ${examples_dir} ]
then
    examples_dir="."
fi
printf "Running examples in ${examples_dir}...\n"
for example in $(ls ${examples_dir}/*.py)
do
    printf "Running example: ${example}\n"
    if [ "mpi" == *"${example}"* ]
    then
        mpiexec -n 2 python ${example}
    else
        python ${example}
    fi
    printf "Example ${example} "
    if [ $? -eq 0 ]
    then
        printf "succeeded.\n"
    else
        printf "failed.\n"
    fi
done
printf "Done running examples!\n" 
#rm -f examples/*.vtu
