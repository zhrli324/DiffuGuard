CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc-per-node=4 --master-port=54321 run.py --data MathVista_MINI --model MMaDA-1
