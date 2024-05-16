USE_FLEXROUND=1 python3 main.py --model_name /mnt/disk4/modelHub/opt-125m/  --bits 4 --group_size -1  --iters 20  --enable_minmax_tuning   --n_samples 32 --lr 1e-4 --disable_amp

# USE_FLEXROUND=1 python3 main.py --model_name /models/opt-125m/  --bits 4 --group_size -1  --iters 200  --enable_minmax_tuning   --n_samples 512 --lr 1e-4 --disable_amp

