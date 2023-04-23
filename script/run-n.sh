#!/bin/bash
sudo nvidia-smi -i 0 -ac 1215,1410

out_file=n-as-x.out.a100
err_file=n-as-x.err.a100
bench="../benchmark/benchmark"

num_iter_large=100
num_iter_small=100

StrNames="drtopk_bitonic,cub,sampleselect,sampleselect-bucket,sampleselect-quick,drtopk_radix,faiss_warp,faiss_block,raft_radix_11bits_extra_pass,grid_select,n_power,k_power,batch,N,k,dist" 
AlgoNames=( drtopk_bitonic cub sampleselect sampleselect-bucket sampleselect-quick drtopk_radix faiss_warp faiss_block raft_radix_11bits_extra_pass grid_select)

ARG=('-w 10 -c ' '-w 10 -c -g ' '-w 10 -c -r 12')
DIST=('Uniform' 'Normal' 'Unfriendly')

echo ${StrNames}>${out_file}
echo echo "Errorlog" > ${err_file}

for i in "${!ARG[@]}"; do	
	arg=${ARG[i]}
	dist=${DIST[i]}
	for bs in 1 100
	do
		for n_power in {11..30}
		do
			for k_power in 5 8 15
			do	
				N=$((2 ** ${n_power}))
				k=$((2 ** ${k_power}))

				if [ $k -ge $N ]; then
					continue
				fi

				# not enough GPU memory on V100
				if [ $(($bs * $N)) -gt $((2 ** 32)) ]; then
					continue
				fi

				for algo in "${AlgoNames[@]}"
				do

				if [ $(($bs * $N)) -gt 1000000 ]; then
					niter=${num_iter_large}
				else
					niter=${num_iter_small}
				fi

				if [ $N -gt 524288 ]; then
					if [[ "$algo" == "faiss_block" || "$algo" == "faiss_warp" ]]; then
							niter=20
					fi
				fi

				cm="${bench} $arg -n $niter $algo $bs $N $k"
				${cm} 2>>${err_file} 1>>${out_file} || echo -n "0.0, " 1>>${out_file}

				done
				echo "${n_power}, ${k_power}, $bs, $N, $k, $dist" 1>>${out_file}
				echo "${n_power}, ${k_power}, $bs, $N, $k, $dist" 
			done
		done
	done 
done

<<'COMMENT'
COMMENT