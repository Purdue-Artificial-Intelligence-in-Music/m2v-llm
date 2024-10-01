#!/bin/bash
#SBATCH --job-name=Batch processing main_cot_model  # Replace with your desired job name
#SBATCH --output=batch_cot_model_output.txt         # Standard output and error log
#SBATCH --nodes=1                                   # Number of nodes
#SBATCH --ntasks=1                                  # Total number of tasks (usually 1 for non-MPI jobs)
#SBATCH --cpus-per-task=32                          # Number of CPUs per task
#SBATCH --gpus-per-node=1                           # Number of GPUs per node
#SBATCH --constraint="I|K"                          # Constraints (specific nodes or features)

if [ "$#" -ne 2 ]; then
    echo "Usage: batch_run.sh <path_to_input_folder> <path_to_output_folder>"
    exit 1
fi

INPUT_FOLDER=$1
OUTPUT_FOLDER=$2
INPUT_FOLDER=$(realpath $INPUT_FOLDER)
OUTPUT_FOLDER=$(realpath $OUTPUT_FOLDER)
VALID_ARGS=1

if [ ! -d "$INPUT_FOLDER" ]; then
    echo "[ERROR] Input folder does not exist."
    VALID_ARGS=0
fi

if [ ! -d "$OUTPUT_FOLDER" ]; then
    if [ -f "$OUTPUT_FOLDER" ]; then
        echo "[ERROR] Output folder is a file."
        VALID_ARGS=0
    else
        echo "[WARN] Output directory didn't exist. Creating it..."
        mkdir $OUTPUT_FOLDER
    fi
fi

if [ $VALID_ARGS -eq 0 ]; then
    exit 1
fi

# Print job details
echo "[METADATA] -----"
echo "Job ID: $SLURM_JOB_ID"
echo "Job Name: $SLURM_JOB_NAME"
echo "User: $USER"
echo "Nodes allocated: $SLURM_JOB_NUM_NODES"
echo "Running on nodes: $SLURM_NODELIST"
echo "Input folder: $1"
echo "Output folder: $2"
echo "[END_METADATA] -----"

# Load modules
module load anaconda
module load ffmpeg

# Set environment variables for optimal performance
export MKL_NUM_THREADS=32
export OPENBLAS_NUM_THREADS=32
export OMP_NUM_THREADS=32

# Set directory
cd /scratch/gilbreth/tnadolsk/m2v-llm/MU-LLaMA/mullama

# For each wav file...
for FILENAME in $(ls $INPUT_FOLDER | grep ".\.wav"); do
    BASENAME=$(echo "$FILENAME" | cut -d '.' -f 1)
    # Run main_cot_model.py with mu_env Conda environment
    echo "[INFO] Processing $(realpath $FILENAME)..."
    /scratch/gilbreth/tnadolsk/m2v-llm/MU-LLaMA/mu_env/bin/python main_cot_model.py --audio_path $INPUT_FOLDER/$FILENAME --output_dir $OUTPUT_FOLDER --output_prefix $BASENAME
    if [ $? -ne 0 ]; then
        echo "[ERROR] Failed to process $FILENAME."
    else
        echo "[SUCCESS] Successfully processed $FILENAME."
    fi
done

# Job done
echo "Job done"