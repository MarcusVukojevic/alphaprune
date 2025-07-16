#!/bin/bash

#SBATCH -p edu-thesis                 # Nome della coda (edu-20h)
#SBATCH --account=giovanni.iacca.tesi
#SBATCH --gres=gpu:a30.24:1              # Richiedi 1 GPU generica   gpu:a30.24:1
#SBATCH --ntasks=1                 # Numero di task
#SBATCH --cpus-per-task=4          # Numero di CPU per task
#SBATCH --mem-per-cpu=7168M        # Memoria per CPU (7 GB per CPU)
#SBATCH -N 1                       # Richiedi 1 nodo
#SBATCH -t 24:00:00                # Tempo massimo di esecuzione (20 ore)
#SBATCH --output=output_%j.txt     # File di output (il %j inserisce il job ID)


# Caricare CUDA (se necessario per l'uso con PyTorch o TensorFlow)
module load CUDA/12.5.0
# Attivare un ambiente virtuale (se presente)
source /home/marcus.vukojevic/.bashrc
conda activate ambiente

# Eseguire lo script Python
python ~/alphaprune/main.py > debug_loss_2.txt

