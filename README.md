
<div align="center">
  
  <div>
  <h1>SLCA++: Unleash the Power of Sequential Fine-tuning for Continual Learning with Pre-training</h1>
  </div>

  <div>
      Gengwei Zhang*&emsp; Liyuan Wang*&emsp; Guoliang Kang&emsp; Ling Chen&emsp; Yunchao Wei
  </div>
  <br/>

</div>


PyTorch code for paper "[SLCA++: Unleash the Power of Sequential Fine-tuning for Continual Learning with Pre-training](https://arxiv.org/abs/2408.08295)", together with the code for our ICCV 2023 paper "[SLCA: Slow Learner with Classifier Alignment for Continual Learning on a Pre-trained Model](https://arxiv.org/abs/2303.05118)". 

## What's new?
[2024.08] We release SLCA++, a parameter-efficient version of SLCA with even better continual performance on fine-grained benchmarks!

## Introduction
In our paper, we present an in-depth analysis of the progressive overfitting problem from the lens of Seq FT. Considering that the overly fast representation learning and the biased classification layer constitute this particular problem, we introduce the advanced Slow Learner with Classifier Alignment (SLCA++) framework to unleash the power of Seq FT, serving as a strong baseline approach for Continual Learning with Pre-training (CLPT). Our approach involves a Slow Learner (SL) to selectively reduce the learning rate of backbone parameters, and a Classifier Alignment (CA) to align the disjoint classification layers in a post-hoc fashion. We further enhance the efficacy of SL with a symmetric cross-entropy loss (SCE), as well as employ a parameter-efficient strategy to implement Seq FT with SLCA++. Across a variety of continual learning scenarios, including class-incremental learning on general datasets like CIFAR-100 and ImageNet-R, fine-grained datasets like CUB-200 and Cars-196, and domain-incremental learning on DomainNet, our approach provides substantial improvements and outperforms state-of-the-art methods by a large margin.



## Requirement
1. torch==1.12.0  
2. torchvision==0.13.0  
3. timm==0.5.4  
4. tqdm  
5. numpy  
6. scipy  
7. quadprog  
8. POT  

## Pre-trained Models
Please download pre-trained ViT-Base models from [MoCo v3](https://drive.google.com/file/d/1bshDu4jEKztZZvwpTVXSAuCsDoXwCkfy/view?usp=share_link) and [ImaegNet-21K](https://drive.google.com/file/d/1PcAOf0tJYs1FVDpj-7lrkSuwXTJXVmuk/view?usp=share_link) and then put or link the pre-trained models to ```SLCA/pretrained```

## Acknowledgement
This repo is heavily based on [PyCIL](https://github.com/G-U-N/PyCIL), many thanks.

## Citation

If you find our codes or paper useful, please consider giving us a star or cite with:  

```
@misc{zhang2024slcaunleashpowersequential,
      title={SLCA++: Unleash the Power of Sequential Fine-tuning for Continual Learning with Pre-training}, 
      author={Zhang, Gengwei and Wang, Liyuan and Kang, Guoliang and Chen, Ling and Wei, Yunchao},
      year={2024},
      eprint={2408.08295},
      archivePrefix={arXiv},
      url={https://arxiv.org/abs/2408.08295}, 
}
```

```
@inproceedings{zhang2023slca,
  title={SLCA: Slow Learner with Classifier Alignment for Continual Learning on a Pre-trained Model},
  author={Zhang, Gengwei and Wang, Liyuan and Kang, Guoliang and Chen, Ling and Wei, Yunchao},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
  year={2023}
}
```
