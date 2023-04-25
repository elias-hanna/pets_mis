#!/bin/bash
# run_mis_analysis

##################################################
##########Execute from data folder################
##################################################

######### Results from cluster ########
reps=10

## Environments
environments=(pusher reacher cartpole halfcheetah)
environments=(pusher reacher)

## considered policy search methods
ims=(random-actions brownian-motion colored-noise-beta-0 colored-noise-beta-1 colored-noise-beta-2 random-policies)
ims=(random-actions brownian-motion colored-noise-beta-0 colored-noise-beta-1 colored-noise-beta-2)


pets_folder=~/src/pets_mis

div_cpt=0
for env in "${environments[@]}"; do
    cd ${env}_pets_results
    echo "Processing following folder"; pwd

    python ${pets_folder}/plot_mis_results.py \
           --init-methods ${ims[*]} -e $env \
           --n-reps $reps
    
    div_cpt=$((div_cpt+1))
    wait
    cd ..
done
