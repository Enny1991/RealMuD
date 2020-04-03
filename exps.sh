#!/usr/bin/env bash

layers=( 4 6 )
causal=( 0 1 )
bn=( 0 1 )
# kernel=( 2 3 )
nfft=( 512 256 )

for f in "${nfft[@]}"
    do
#    for k in "${kernel[@]}"
#    do
        for b in "${bn[@]}"
        do
            for c in "${causal[@]}"
            do
                for l in "${layers[@]}"
                do
                    sed 's/%args%/'${f}' '${b}' '${c}' '${l}'/g' < mud_cuda10.sh | bsub
                done
#            done
        done
    done
done