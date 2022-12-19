lrs=( "0.0001" "0.001" "0.01" )
wds=( "0.0005" "0.0001" "0.005" )

for lr in ${lrs[@]}
do
    for wd in ${wds[@]}
    do
	echo "python ./transformer_tuning.py --workers 4 --batch_size 64 --ngd --lr ${lr} --weighted_decay ${wd} --epoch 5"
        python ./transformer_tuning.py --workers 4 --batch_size 64 --ngd --lr ${lr} --weight_decay ${wd} --epoch 5
    done
done