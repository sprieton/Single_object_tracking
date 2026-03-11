#!/bin/bash

# Archivo donde se guardarán todos los resultados
OUTPUT_FILE="parameter_sweep_results.txt"
echo "=== Parameter Sweep Results ===" > $OUTPUT_FILE
echo "sigma_x sigma_y sigma_w sigma_h sigma_vx sigma_vy sigma_vw sigma_vh alpha speed_noise_factor JI avg_time_per_frame" >> $OUTPUT_FILE

# Definir arrays con valores a probar
SIGMA_VALUES=("0.25 0.25 0.01 0.01 0.01 0.01 0.001 0.001" "0.3 0.3 0.02 0.02 0.01 0.01 0.001 0.001" "0.2 0.2 0.01 0.01 0.01 0.01 0.001 0.001")
ALPHA_VALUES=(15 20 25)
SPEED_NOISE_VALUES=(0.4 0.5 0.6)

# Número de partículas y repeticiones
N_PARTICLES=300
REPETITIONS=4

# Loop sobre todas las combinaciones
for sigma in "${SIGMA_VALUES[@]}"; do
    for alpha in "${ALPHA_VALUES[@]}"; do
        for speed_noise in "${SPEED_NOISE_VALUES[@]}"; do
            echo "Running with sigma=$sigma alpha=$alpha speed_noise_factor=$speed_noise"
            
            # Ejecutar el script de evaluación y capturar la última línea con el JI promedio
            RESULT=$(python3 evaluateSystem.py --N $N_PARTICLES --repetitions $REPETITIONS \
                     --sigma $sigma --alpha $alpha --speed_noise_factor $speed_noise | \
                     tail -n 3 | head -n 1)

            # Guardar los parámetros y el resultado en el fichero
            echo "$sigma $alpha $speed_noise $RESULT" >> $OUTPUT_FILE
        done
    done
done

echo "Parameter sweep finished. Results saved in $OUTPUT_FILE"
