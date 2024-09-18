#!/bin/sh

# dataset_name="thumos14"
dataset_name="fineaction"
echo ${dataset_name}

# python convert_result_for_tal.py --model_name flash --dataset_name ${dataset_name} --time_inst_ver 1 --anno_ver 0 
# python convert_result_for_tal.py --model_name flash --dataset_name ${dataset_name} --time_inst_ver 1 --anno_ver 1 
# python convert_result_for_tal.py --model_name flash --dataset_name ${dataset_name} --time_inst_ver 1 --anno_ver 2 

python convert_result_for_tal.py --model_name pro --dataset_name ${dataset_name} --time_inst_ver 1 --anno_ver 0
# python convert_result_for_tal.py --model_name pro --dataset_name ${dataset_name} --time_inst_ver 1 --anno_ver 1
# python convert_result_for_tal.py --model_name pro --dataset_name ${dataset_name} --time_inst_ver 1 --anno_ver 2

cp tal_output/${dataset_name}/*/*/*/*.json tal_output_in_json/${dataset_name}