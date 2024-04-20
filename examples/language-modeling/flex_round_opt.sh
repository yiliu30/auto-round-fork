USE_FLEXROUND=1 python3 main.py --model_name /models/opt-125m/  --bits 8 --group_size -1  --iters 500 --enable_minmax_tuning   --n_samples 128 --lr 3e-5 --disable_amp --eval_bs 8 &> ./_flex_log/_opt-125m_8bits_iter500_lr_3e-5_nsample_128_ppl
ONLY_PPL=1 USE_FLEXROUND=1 python3 main.py --model_name /models/opt-125m/  --bits 8 --group_size -1  --iters 500 --enable_minmax_tuning   --n_samples 128 --lr 3e-5 --disable_amp --eval_bs 8 &> ./_flex_log/_ada_opt-125m_8bits_iter500_lr_3e-5_nsample_128_ppl
# fine-tuned
# epoch 10
# lr 3e-5
# bs 8
# recon
# per-tensor asymmetric scheme.
# iters 500
# lr 3r-5
# bs 32
# samples 128
# paper ppl 21.43

# opt-1.3b 
# fine-tuned
# epoch 10
# lr 3e-6
# bs 4
# recon
# per-tensor asymmetric scheme.
# iters 500
# lr 7e-6
# bs 16
# samples 128
# paper ppl 21.43