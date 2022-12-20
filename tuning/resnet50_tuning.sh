alphas=( "0.99" "0.9" "0.8" )
gammas=( "0.75" "0.85" "0.95" )

for alpha in ${alphas[@]}
do
    for gamma in ${gammas[@]}
    do
	echo "python ./resnet50_tuning.py --workers 4 --bs 256 --ngd --alpha ${alpha} --gamma ${gamma} --epoch 5"
        python ./resnet50_tuning.py --workers 4 --bs 256 --ngd --alpha ${alpha} --gamma ${gamma} --epoch 5
    done
done