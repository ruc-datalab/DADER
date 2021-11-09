#!/bin/bash
# for similar domains and different domains
srcd=(wa1 ab ds da dzy fz ri ri ia ia b2 b2)
tgtd=(ab wa1 da ds fz dzy ab wa1 da ds fz dzy)
ne=40
cudaid=2

for seed in 3000 1000 42 10 0
do 
    for i in 0 1 2 3 8
    do
        CUDA_VISIBLE_DEVICES=$cudaid python ../main/main_mmd.py \
            --src ${srcd[$i]} \
            --tgt ${tgtd[$i]} \
            --max_seq_length 128 \
            --train_seed $seed \
            --num_epochs $ne \
            --beta 0.1
    done
done

for seed in 3000 1000 42 10 0
do 
    for i in 4 5 9 10 11
    do
        CUDA_VISIBLE_DEVICES=$cudaid python ../main/main_mmd.py \
            --src ${srcd[$i]} \
            --tgt ${tgtd[$i]} \
            --max_seq_length 128 \
            --train_seed $seed \
            --num_epochs $ne \
            --beta 0.01
    done
done

for seed in 3000 1000 42 10 0
do 
    for i in 6 7
    do
        CUDA_VISIBLE_DEVICES=$cudaid python ../main/main_mmd.py \
            --src ${srcd[$i]} \
            --tgt ${tgtd[$i]} \
            --max_seq_length 128 \
            --train_seed $seed \
            --num_epochs $ne \
            --beta 0.001
    done
done

srcd=(computers cameras shoes computers cameras computers)
tgtd=(watches watches watches shoes shoes cameras)
cudaid=1
ne=40

for seed in 3000 1000 42 10 0    
do 
    for i in 0 1 2 3 4 5 6
    do 
        CUDA_VISIBLE_DEVICES=$cudaid python ../main/main_mmd.py \
            --src ${srcd[$i]} \
            --tgt ${tgtd[$i]} \
            --max_seq_length 256 \
            --train_seed $seed \
            --num_epochs $ne \
            --beta 0.01
    done
done
