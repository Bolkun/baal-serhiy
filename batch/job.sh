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

source lib.sh

#remove_new_environment pyjob_baal
create_or_reuse_environment

# experiments
echo "Experiments"
cd /home/sebo742e/baal-serhiy/experiments

echo "Start"
# Example to use only cifar10 original data
#python vgg_mcdropout_cifar10_original.py --epoch 100
# Example to use cifar10 original + augmented data
#python vgg_mcdropout_cifar10_augment.py --epoch 100
# Example to use only cifar10 original + augmented data
#python vgg_mcdropout_cifar10_org+aug_1.py --epoch 100
python vgg_mcdropout_cifar10_org+aug_2.py --epoch 1
echo "End"
