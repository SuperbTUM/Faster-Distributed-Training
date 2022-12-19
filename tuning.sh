for lr in {"0.0001", "0.001", "0.01"}
do 
    for wd in { "0.0005", "0.0001", "0.005"}
    do 
        python ./transformer_test.py --workers 4 --batch_size 64 --ngd --lr ${lr} --weight_decay ${wd}
    done 
done 