# for similar domains and different domains
srcd=(wa1 ab ds da dzy fz ri ri ia ia b2 b2)
tgtd=(ab wa1 da ds fz dzy ab wa1 da ds fz dzy)
ne=20
cudaid=0

for seed in 3000 1000 42 10 0    
do 
    for i in 0 1 2 3 4 5
    do
            CUDA_VISIBLE_DEVICES=$cudaid python ../main/main_noda.py \
                --pretrain \
                --src ${srcd[$i]} \
                --tgt ${tgtd[$i]} \
                --srcfix ""\
                --tgtfix ""\
                --max_seq_length 128 \
                --train_seed $seed \
                --pre_epochs $ne
    done
done

for seed in 3000 1000 42 10 0    
do 
    for i in 6 7 8 9 10 11
    do
            CUDA_VISIBLE_DEVICES=$cudaid python ../main/main_noda.py \
                --pretrain \
                --src ${srcd[$i]} \
                --tgt ${tgtd[$i]} \
                --srcfix ""\
                --tgtfix ""\
                --max_seq_length 128 \
                --train_seed $seed \
                --pre_epochs $ne
    done
done

# for WDC datasets
srcd=(computers cameras shoes computers cameras computers)
tgtd=(watches watches watches shoes shoes cameras)
cudaid=0
ne=20

for seed in 3000 1000 42 10 0    
do 
    for i in 0 1 2 3 4 5 6
    do 
            CUDA_VISIBLE_DEVICES=$cudaid python ../main/main_noda.py \
                --pretrain \
                --src ${srcd[$i]} \
                --tgt ${tgtd[$i]} \
                --srcfix ""\
                --tgtfix ""\
                --max_seq_length 128 \
                --train_seed $seed \
                --pre_epochs $ne

            CUDA_VISIBLE_DEVICES=$cudaid python ../main/main_noda.py \
                --pretrain \
                --src ${tgtd[$i]} \
                --tgt ${srcd[$i]} \
                --srcfix ""\
                --tgtfix ""\
                --max_seq_length 128 \
                --train_seed $seed \
                --pre_epochs $ne

    done
done
