CUDA_VISIBLE_DEVICES=0 python3 main.py --config=exps/slcapp/slcapp_cifar_lora.json & 
CUDA_VISIBLE_DEVICES=1 python3 main.py --config=exps/slcapp/slcapp_imgnetr_lora.json &
CUDA_VISIBLE_DEVICES=2 python3 main.py --config=exps/slcapp/slcapp_cub_lora.json & 
CUDA_VISIBLE_DEVICES=3 python3 main.py --config=exps/slcapp/slcapp_cars_lora.json &

CUDA_VISIBLE_DEVICES=0 python3 main.py --config=exps/slcapp/slcapp_cifar_lora_mocov3.json &
CUDA_VISIBLE_DEVICES=1 python3 main.py --config=exps/slcapp/slcapp_imgnetr_lora_mocov3.json &
CUDA_VISIBLE_DEVICES=2 python3 main.py --config=exps/slcapp/slcapp_cub_lora_mocov3.json &
CUDA_VISIBLE_DEVICES=3 python3 main.py --config=exps/slcapp/slcapp_cars_lora_mocov3.json &