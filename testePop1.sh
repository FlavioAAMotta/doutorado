pop="Pop1"
declare -a classifiers=("SVMR" "RF" "KNN" "DCT" "LR" "SVML" "SVMS" "HV" "SV" "SV-Grid")
# declare -a classifiers=("SV-Grid" "SV")
declare -a windows=("1" "4")
declare -a steps=("1" "4")
declare -a thresholds=("0.5")
declare -a withVolume=("False")
timestamp=$(date +%y%m%d-%H%M)
folder="results/${pop}/${timestamp}"
mkdir -p "$folder"

# Gerando a lista de comandos
commands=()

for i in "${classifiers[@]}"; do
    for j in "${windows[@]}"; do
        for k in "${steps[@]}"; do
            if [ "$k" -gt "$j" ]; then
                continue
            fi
            for l in "${thresholds[@]}"; do
                for m in "${withVolume[@]}"; do
                    cmd="python main.py $pop $i $j $k $l $m >> $folder/${i}_${j}_${k}_${l}_${m}.txt"
                    commands+=("$cmd")
                done
            done
        done
    done
done

# Executando os comandos em paralelo
total_commands=${#commands[@]}
command_number=0

for cmd in "${commands[@]}"; do
    ((command_number++))
    bash -c "$cmd" &
    # Limitar o número de processos em paralelo
    if (( $command_number % 3 == 0 )); then
        wait # Aguarda os três comandos anteriores terminarem
    fi
    echo "Comando $command_number de $total_commands iniciado."
done

wait # Aguarda os últimos comandos terminarem
echo "Todos os comandos foram executados."

# pop="Pop1"
# declare -a classifiers=("SV-Grid" "SV")
# declare -a windows=("1" "4")
# declare -a steps=("1" "4")
# declare -a thresholds=("0.5")
# declare -a withVolume=("False")
# timestamp=$(date +%y%m%d-%H%M)
# folder="results/${pop}/${timestamp}"
# mkdir -p "$folder"

# # Gerando a lista de comandos
# commands=()

# for i in "${classifiers[@]}"; do
#     for j in "${windows[@]}"; do
#         for k in "${steps[@]}"; do
#             if [ "$k" -gt "$j" ]; then
#                 continue
#             fi
#             for l in "${thresholds[@]}"; do
#                 for m in "${withVolume[@]}"; do
#                     cmd="python main.py $pop $i $j $k $l $m >> $folder/${i}_${j}_${k}_${l}_${m}.txt"
#                     commands+=("$cmd")
#                 done
#             done
#         done
#     done
# done

# # Executando os comandos sequencialmente
# total_commands=${#commands[@]}
# command_number=0

# for cmd in "${commands[@]}"; do
#     ((command_number++))
#     bash -c "$cmd"
#     echo "Comando $command_number de $total_commands completado."
# done

# echo "Todos os comandos foram executados."