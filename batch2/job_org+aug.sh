#!/bin/bash
                           
#SBATCH --no-requeue
#SBATCH --partition=alpha
#SBATCH --nodes=1                   
#SBATCH --gres=gpu:1
#SBATCH --mem=8000
#SBATCH --mincpus=1
#SBATCH --time=6-23:59:59                            
#SBATCH --job-name=baal
#SBATCH --mail-type=ALL
#SBATCH --mail-user=serhij16@live.de
#SBATCH --output=output-jobId-%j.out
#SBATCH --error=error-jobId-%j.out

module --force purge                          				
module load modenv/hiera CUDA/11.7.0 GCCcore/11.3.0 Python/3.10.4

source lib_org+aug.sh

#remove_new_environment pyjob_baal
create_or_reuse_environment

# experiments x3 for every run
echo "Experiments"
cd /home/sebo742e/baal-serhiy2/experiments

echo "Start"

# cifar10 original + augmented data
#python vgg_mcdropout_cifar10_org+aug.py --epoch 50 --initial_pool 100 --learning_epoch 5
#python vgg_mcdropout_cifar10_org+aug.py --epoch 50 --initial_pool 100 --learning_epoch 5 --heuristic "entropy"
python vgg_mcdropout_cifar10_org+aug.py --epoch 50 --initial_pool 100 --learning_epoch 5 --pretrained 1


echo "End"
