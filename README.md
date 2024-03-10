## Default Branch

Updated branch is ***dev***. Use following command to change branch.

`git swich dev`

## Train and Quantize the Model

Run the following command to train and quantize the AI model. Results of this command ***quantized_model.h5*** file is created.

`python3 train_and_quantize.py`

## Compile the Model

arch.json file is required for compile the model. First of all you should create the json file with ***DPU_CONFIG*** name under the arch_files location.

Example arch file:
DPU_C1_B500.json

After the json file creation step, you should run the ***compile_model.sh script***. This model take ***DPU_CONFIG*** parameter.

Example compilation script:

`./compile_model.sh DPU_C1_B500`

## Run Vitis AI Model on Kria KV260

Run ***deployment.py*** file for the run Vitis AI model on Kria KV260 boards. This python code take ***DPU_CONFIG*** parameter.

***NOTE:*** arch json file name and ompile_model.sh script parameter and deployment.py script parameter should be same for correct deployment.

Example script for run model on Kria KV260

`python3 deployment.py DPU_C1_B500`





