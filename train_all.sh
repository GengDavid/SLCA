CUDA_VISIBLE_DEVICES=0,1 python3 main.py --config=exps/slca_cifar.json
CUDA_VISIBLE_DEVICES=0,1 python3 main.py --config=exps/slca_imgnetr.json
CUDA_VISIBLE_DEVICES=0,1 python3 main.py --config=exps/slca_cub.json
CUDA_VISIBLE_DEVICES=0,1 python3 main.py --config=exps/slca_cars.json

CUDA_VISIBLE_DEVICES=0,1 python3 main.py --config=exps/slca_cifar_mocov3.json
CUDA_VISIBLE_DEVICES=0,1 python3 main.py --config=exps/slca_imgnetr_mocov3.json
CUDA_VISIBLE_DEVICES=0,1 python3 main.py --config=exps/slca_cub_mocov3.json
CUDA_VISIBLE_DEVICES=0,1 python3 main.py --config=exps/slca_cars_mocov3.json

