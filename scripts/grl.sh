#!/bin/bash

srcd=(wa1 ab ds da dzy fz ri ri ia ia b2 b2)
tgtd=(ab wa1 da ds fz dzy ab wa1 da ds fz dzy)
ne=40
cudaid=3

for seed in 42 10 0 3000 1000 
do 
    for i in 1 5 10
    do
        CUDA_VISIBLE_DEVICES=$cudaid python ../main/main_grl.py \
            --src ${srcd[$i]} \
            --tgt ${tgtd[$i]} \
            --max_seq_length 128 \
            --train_seed $seed \
            --num_epochs $ne \
            --alpha 0.01 \
            --beta 0.1
            
    done
done

for seed in 42 10 0 3000 1000 
do 
    for i in 0 2 3 6 7 8 9 11
    do
        CUDA_VISIBLE_DEVICES=$cudaid python ../main/main_grl.py \
            --src ${srcd[$i]} \
            --tgt ${tgtd[$i]} \
            --max_seq_length 128 \
            --train_seed $seed \
            --num_epochs $ne \
            --alpha 0.001 \
            --beta 0.1
            
    done
done

for seed in 42 10 0 3000 1000 
do 
    for i in 4
    do
        CUDA_VISIBLE_DEVICES=$cudaid python ../main/main_grl.py \
            --src ${srcd[$i]} \
            --tgt ${tgtd[$i]} \
            --max_seq_length 128 \
            --train_seed $seed \
            --num_epochs $ne \
            --alpha 0.001 \
            --beta 0.01
            
    done
done

srcd=(computers cameras shoes computers cameras computers)
tgtd=(watches watches watches shoes shoes cameras)
cudaid=0
ne=40

for seed in 3000 1000 42 10 0    
do 
    for i in 0 1 2 3 4 5 6
    do 
        CUDA_VISIBLE_DEVICES=$cudaid python ../main/main_grl.py \
            --src ${srcd[$i]} \
            --tgt ${tgtd[$i]} \
            --max_seq_length 256 \
            --train_seed $seed \
            --num_epochs $ne \
            --alpha 0.001 \
            --beta 0.1
    done
done