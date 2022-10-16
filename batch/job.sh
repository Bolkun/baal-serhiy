#!/bin/bash
                           
#SBATCH --no-requeue
#SBATCH --partition=alpha
#SBATCH --nodes=1                   
#SBATCH --gres=gpu:1
#SBATCH --mem=8000
#SBATCH --mincpus=1
#SBATCH --time=23:59:00                             
#SBATCH --job-name=baal
#SBATCH --mail-type=ALL
#SBATCH --mail-user=serhij16@live.de
#SBATCH --output=output-jobId-%j.out
#SBATCH --error=error-jobId-%j.out

module --force purge                          				
module load modenv/hiera CUDA/11.3.1 GCC/11.2.0 Python/3.9.6

source lib.sh

remove_new_environment pyjob_baal
create_or_reuse_environment

# experiments
echo "Experiments"
cd /home/sebo742e/baal-serhiy/experiments
echo "Start"
python vgg_mcdropout_cifar10.py --epoch 1
