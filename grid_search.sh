#!/bin/bash

# Archivo de salida CSV
OUTPUT_FILE="parameter_sweep_results.csv"
echo "noise_beta,ellipse_area_ratio,mcmc_expl_fact,JI,avg_time_per_frame" > $OUTPUT_FILE

# Valores a probar
NOISE_BETA=(1.5 1.55 1.45)
ELIPSE_RATIO=(0.5)

MCMC_FACT=(1.2 1.3)

# Número de partículas y repeticiones
N_PARTICLES=100
REPETITIONS=2

# Loop sobre todas las combinaciones
for noise_beta in "${NOISE_BETA[@]}"; do
    for ellipse_area_ratio in "${ELIPSE_RATIO[@]}"; do
        for mcmc_expl_fact in "${MCMC_FACT[@]}"; do
            echo "Running with noise_beta=$noise_beta ellipse_area_ratio=$ellipse_area_ratio mcmc_expl_fact=$mcmc_expl_fact"

            # Ejecutar el script de evaluación
            OUTPUT=$(python3 evaluateSystem.py --N $N_PARTICLES --repetitions $REPETITIONS \
                     --noise_beta $noise_beta --ellipse_area_ratio $ellipse_area_ratio --mcmc_expl_fact $mcmc_expl_fact)

            # Extraer el JI promedio final y avg_time_per_frame
            JI=$(echo "$OUTPUT" | grep "Total average Results are JI=" | awk -F'=' '{print $2}' | tr -d ' ')
            AVG_TIME=$(echo "$OUTPUT" | grep "Results for video" | tail -n4 | awk -F'=' '{sum += $3} END {print sum/NR}')

            # Guardar en CSV
            echo "$noise_beta,$ellipse_area_ratio,$mcmc_expl_fact,$JI,$AVG_TIME" >> $OUTPUT_FILE
        done
    done
done

echo "Parameter sweep finished. Results saved in $OUTPUT_FILE"