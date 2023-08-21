

#!/bin/bash 

# echo "Starting script" 
# if [ "$#" -lt 1 ]; then # exit if called with no arguments 
#     echo "Usage: bash $0  <CONDA ENVIRONMENT NAME>"
#     exit 1
# fi 


# CONDA_ENV_DIR="/Users/gadmohamed/miniforge3/envs"
# CONDA_ENV="$1"

# CODE_PATH="src/main.py"


# source /opt/anaconda3/etc/profile.d/conda.sh
# conda init zsh
# conda activate $CONDA_ENV 

# REQUIREMENTS_FILE="src/requirements.txt"
# if [ -f "$REQUIREMENTS_FILE" ]; then
#     echo "Found $REQUIREMENTS_FILE file. Installing dependencies..."
#     pip install -r "$REQUIREMENTS_FILE"
#     echo "Done installing dependencies." 
# else
#     echo "No $REQUIREMENTS_FILE file found."
# fi


echo "Begin experiments!" 


# python edit_config.py CIFAR_balance_conf algorithm fedmd  aug False select False compress False #fedmd
# python CIFAR_Balanced.py

python edit_config.py CIFAR_balance_conf aug True #fedakd
python CIFAR_Balanced.py

python edit_config.py CIFAR_balance_conf compress True #cfedakd
python CIFAR_Balanced.py

python edit_config.py CIFAR_balance_conf compress False select True #sfedakd
python CIFAR_Balanced.py

python edit_config.py CIFAR_balance_conf aug False #sfedmd
python CIFAR_Balanced.py

python edit_config.py CIFAR_balance_conf algorithm fedavg #fedavg
python CIFAR_Balanced.py


echo "Done experiments!"

# datasets=("cifar10" "mnist")
# # datasets=("mnist")

# learning_algorithms=("fedavg" "fedakd" "fedsgd")
# # learning_algorithms=("central")
# dp_types=("dp" "adv_cmp" "rdp")
# dp_epsilon_values=(0.1 1 10 100 1000 2000)
# learning_rates=(0.15)

# # private learning
# for dataset in "${datasets[@]}"; do
#     for learning_algorithm in "${learning_algorithms[@]}"; do
#         for dp_type in "${dp_types[@]}"; do
#             for dp_epsilon in "${dp_epsilon_values[@]}"; do
#                 for learning_rate in "${learning_rates[@]}"; do
#                     python "$CODE_PATH" "$dataset" --learning_algorithm "$learning_algorithm" --use_dp --rounds 20 --local_epochs 1 --target_model='nn' --dp_epsilon $dp_epsilon --dp_type $dp_type --lr $learning_rate
#                 done
#             done
#         done
#     done
# done

# Non private learning
# for dataset in "${datasets[@]}"; do
#     for learning_algorithm in "${learning_algorithms[@]}"; do
#         python "$CODE_PATH" "$dataset" --learning_algorithm "$learning_algorithm" --rounds 40 --local_epochs 1 --target_model='nn' --lr 0.01
#         python "$CODE_PATH" "$dataset" --learning_algorithm "$learning_algorithm" --rounds 40 --local_epochs 1 --target_model='nn' --lr 0.001
#     done
# done


# python $CODE_PATH $DATASET --learning_algorithm 'fedprox' --rounds 10  --local_epochs 1  --target_model='nn' --dp_epsilon 100 --dp_type 'dp'
# python $CODE_PATH $DATASET --learning_algorithm 'fedsgd' --rounds 10 --use_dp --local_epochs 1  --target_model='nn' --dp_epsilon 100 --dp_type 'dp'

# python $CODE_PATH $DATASET --learning_algorithm 'fedprox' --rounds 10 --local_epochs 1  --target_model='nn' 
# python $CODE_PATH $DATASET --learning_algorithm 'fedsgd' --rounds 10 --local_epochs 1  --target_model='nn'

# python $CODE_PATH $DATASET --learning_algorithm 'local' --local_epochs 40  --target_model='nn' 
# python $CODE_PATH $DATASET --learning_algorithm 'local' --local_epochs 40  --use_dp --target_model='nn'  --dp_epsilon 100 --dp_type 'dp'
# python $CODE_PATH $DATASET --learning_algorithm 'local' --local_epochs 40  --use_dp --lr 0.1 --target_model='nn' --dp_epsilon 100 --dp_type 'dp'

# python $CODE_PATH $DATASET --learning_algorithm 'fedavg' --use_dp --rounds 20 --local_epochs 1 --target_model='nn' --dp_epsilon 1 --dp_type 'dp' --lr 0.1
# python $CODE_PATH $DATASET --learning_algorithm 'fedavg' --use_dp --rounds 20 --local_epochs 1 --target_model='nn' --dp_epsilon 10 --dp_type 'dp' --lr 0.1
# python $CODE_PATH $DATASET --learning_algorithm 'fedavg' --use_dp --rounds 20 --local_epochs 1 --target_model='nn' --dp_epsilon 100 --dp_type 'dp' --lr 0.1
# python $CODE_PATH $DATASET --learning_algorithm 'fedavg' --use_dp --rounds 20 --local_epochs 1 --target_model='nn' --dp_epsilon 1000 --dp_type 'dp' --lr 0.1
# python $CODE_PATH $DATASET --learning_algorithm 'fedavg' --use_dp --rounds 20 --local_epochs 1 --target_model='nn' --dp_epsilon 1 --dp_type 'rdp' --lr 0.1
# python $CODE_PATH $DATASET --learning_algorithm 'fedavg' --use_dp --rounds 20 --local_epochs 1 --target_model='nn' --dp_epsilon 10 --dp_type 'rdp' --lr 0.1
# python $CODE_PATH $DATASET --learning_algorithm 'fedavg' --use_dp --rounds 20 --local_epochs 1 --target_model='nn' --dp_epsilon 100 --dp_type 'rdp' --lr 0.1
# python $CODE_PATH $DATASET --learning_algorithm 'fedavg' --use_dp --rounds 20 --local_epochs 1 --target_model='nn' --dp_epsilon 1000 --dp_type 'rdp' --lr 0.1

# python $CODE_PATH $DATASET --learning_algorithm 'fedavg' --use_dp --rounds 20 --local_epochs 1 --target_model='nn' --dp_epsilon 1 --dp_type 'dp' --lr 0.01
# python $CODE_PATH $DATASET --learning_algorithm 'fedavg' --use_dp --rounds 20 --local_epochs 1 --target_model='nn' --dp_epsilon 10 --dp_type 'dp' --lr 0.01
# python $CODE_PATH $DATASET --learning_algorithm 'fedavg' --use_dp --rounds 20 --local_epochs 1 --target_model='nn' --dp_epsilon 100 --dp_type 'dp' --lr 0.01
# python $CODE_PATH $DATASET --learning_algorithm 'fedavg' --use_dp --rounds 20 --local_epochs 1 --target_model='nn' --dp_epsilon 1000 --dp_type 'dp' --lr 0.01
# python $CODE_PATH $DATASET --learning_algorithm 'fedavg' --use_dp --rounds 20 --local_epochs 1 --target_model='nn' --dp_epsilon 1 --dp_type 'rdp' --lr 0.01
# python $CODE_PATH $DATASET --learning_algorithm 'fedavg' --use_dp --rounds 20 --local_epochs 1 --target_model='nn' --dp_epsilon 10 --dp_type 'rdp' --lr 0.01
# python $CODE_PATH $DATASET --learning_algorithm 'fedavg' --use_dp --rounds 20 --local_epochs 1 --target_model='nn' --dp_epsilon 100 --dp_type 'rdp' --lr 0.01
# python $CODE_PATH $DATASET --learning_algorithm 'fedavg' --use_dp --rounds 20 --local_epochs 1 --target_model='nn' --dp_epsilon 1000 --dp_type 'rdp' --lr 0.01
# python $CODE_PATH $DATASET --learning_algorithm 'fedavg'  --rounds 20  --local_epochs 1  --target_model='nn' --lr 0.01
# python $CODE_PATH $DATASET --learning_algorithm 'fedavg'  --rounds 20  --local_epochs 1  --target_model='nn' --lr 0.001
# python $CODE_PATH $DATASET --learning_algorithm 'fedavg'  --rounds 20  --local_epochs 1  --target_model='nn' --lr 0.0001



# python $CODE_PATH $DATASET --id 10 --learning_algorithm 'central' --local_epochs 20  --target_model='nn' --dp_epsilon 1 --dp_type 'dp' --lr 0.01
# python $CODE_PATH $DATASET --id 11 --learning_algorithm 'central' --use_dp --local_epochs 20  --target_model='nn' --dp_epsilon 10 --dp_type 'dp' --lr 0.01
# python $CODE_PATH $DATASET --id 12 --learning_algorithm 'central' --use_dp --local_epochs 20  --target_model='nn' --dp_epsilon 100 --dp_type 'dp' --lr 0.01
# python $CODE_PATH $DATASET --id 12 --learning_algorithm 'central' --use_dp --local_epochs 20  --target_model='nn' --dp_epsilon 1000 --dp_type 'dp' --lr 0.01

# python $CODE_PATH $DATASET --id 10 --learning_algorithm 'central' --local_epochs 20  --target_model='nn' --dp_epsilon 1 --dp_type 'dp' --lr 0.001
# python $CODE_PATH $DATASET --id 11 --learning_algorithm 'central' --use_dp --local_epochs 20  --target_model='nn' --dp_epsilon 10 --dp_type 'dp' --lr 0.001
# python $CODE_PATH $DATASET --id 12 --learning_algorithm 'central' --use_dp --local_epochs 20  --target_model='nn' --dp_epsilon 100 --dp_type 'dp' --lr 0.001
# python $CODE_PATH $DATASET --id 12 --learning_algorithm 'central' --use_dp --local_epochs 20  --target_model='nn' --dp_epsilon 1000 --dp_type 'dp' --lr 0.001

# python $CODE_PATH $DATASET --id 10 --learning_algorithm 'central' --local_epochs 20  --target_model='nn' --lr 0.01
# python $CODE_PATH $DATASET --id 11 --learning_algorithm 'central' --use_dp --local_epochs 20  --target_model='nn' --dp_epsilon 10 --dp_type 'rdp' --lr 0.01
# python $CODE_PATH $DATASET --id 12 --learning_algorithm 'central' --use_dp --local_epochs 20  --target_model='nn' --dp_epsilon 100 --dp_type 'rdp' --lr 0.01
# python $CODE_PATH $DATASET --id 12 --learning_algorithm 'central' --use_dp --local_epochs 20  --target_model='nn' --dp_epsilon 1000 --dp_type 'rdp' --lr 0.01

# python $CODE_PATH $DATASET --id 10 --learning_algorithm 'central' --local_epochs 20  --target_model='nn' --lr 0.001
# python $CODE_PATH $DATASET --id 11 --learning_algorithm 'central' --use_dp --local_epochs 20  --target_model='nn' --dp_epsilon 10 --dp_type 'rdp' --lr 0.001
# python $CODE_PATH $DATASET --id 12 --learning_algorithm 'central' --use_dp --local_epochs 20  --target_model='nn' --dp_epsilon 100 --dp_type 'rdp' --lr 0.001
# python $CODE_PATH $DATASET --id 12 --learning_algorithm 'central' --use_dp --local_epochs 20  --target_model='nn' --dp_epsilon 1000 --dp_type 'rdp' --lr 0.001


# python $CODE_PATH $DATASET --learning_algorithm 'central' --use_dp  --local_epochs 10  --target_model='nn' --dp_epsilon 10 --dp_type 'dp' --lr 0.001
# python $CODE_PATH $DATASET --learning_algorithm 'central' --local_epochs 10  --target_model='nn' 

# python $CODE_PATH $DATASET --learning_algorithm 'fedakd' --rounds 30 --local_epochs 1  --target_model='nn' --dp_epsilon 1  --lr 0.01
# python $CODE_PATH $DATASET --learning_algorithm 'fedakd' --rounds 20 --local_epochs 1  --target_model='nn' --dp_epsilon 1  --lr 0.001

# python $CODE_PATH $DATASET --learning_algorithm 'fedakd' --use_dp --rounds 20 --local_epochs 1  --target_model='nn' --dp_epsilon 1 --dp_type 'dp' --lr 0.01
# python $CODE_PATH $DATASET --learning_algorithm 'fedakd' --use_dp --rounds 20 --local_epochs 1  --target_model='nn' --dp_epsilon 10 --dp_type 'dp' --lr 0.01
# python $CODE_PATH $DATASET --learning_algorithm 'fedakd' --use_dp --rounds 20 --local_epochs 1  --target_model='nn' --dp_epsilon 100 --dp_type 'dp' --lr 0.01
# python $CODE_PATH $DATASET --learning_algorithm 'fedakd' --use_dp --rounds 20 --local_epochs 1  --target_model='nn' --dp_epsilon 1000 --dp_type 'dp' --lr 0.01

# python $CODE_PATH $DATASET --learning_algorithm 'fedakd' --use_dp --rounds 20 --local_epochs 1  --target_model='nn' --dp_epsilon 1 --dp_type 'rdp' --lr 0.01
# python $CODE_PATH $DATASET --learning_algorithm 'fedakd' --use_dp --rounds 20 --local_epochs 1  --target_model='nn' --dp_epsilon 10 --dp_type 'rdp' --lr 0.01
# python $CODE_PATH $DATASET --learning_algorithm 'fedakd' --use_dp --rounds 20 --local_epochs 1  --target_model='nn' --dp_epsilon 100 --dp_type 'rdp' --lr 0.01
# python $CODE_PATH $DATASET --learning_algorithm 'fedakd' --use_dp --rounds 20 --local_epochs 1  --target_model='nn' --dp_epsilon 1000 --dp_type 'rdp' --lr 0.01
# python $CODE_PATH $DATASET --learning_algorithm 'fedakd' --use_dp --rounds 100 --local_epochs 1  --target_model='nn' --dp_epsilon 100 --dp_type 'rdp' --lr 0.01
# python $CODE_PATH $DATASET --learning_algorithm 'fedakd' --use_dp --rounds 100 --local_epochs 1  --target_model='nn' --dp_epsilon 100 --dp_type 'dp' --lr 0.01
# python $CODE_PATH $DATASET --learning_algorithm 'fedakd' --rounds 20 --local_epochs 1  --target_model='nn' 

