
<div align="center">
  
  <div>
  <h1>SLCA: Slow Learner with Classifier Alignment for Continual Learning on a Pre-trained Model</h1>
  </div>

  <div>
      Gengwei Zhang*&emsp; Liyuan Wang*&emsp; Guoliang Kang&emsp; Ling Chen&emsp; Yunchao Wei
  </div>
  <br/>

</div>


PyTorch code for ICCV 2023 paper "[SLCA: Slow Learner with Classifier Alignment for Continual Learning on a Pre-trained Model](https://arxiv.org/abs/2303.05118)".

In this work, we present an extensive analysis for continual learning on a pre-trained model (CLPM), and attribute the key challenge to a progressive overfitting problem. 
Observing that selectively reducing the learning rate can almost resolve this issue in the representation layer, we propose a simple but extremely effective approach named 
**Slow Learner with Classifier Alignment (SLCA)**, 
which further improves the classification layer by modelling the class-wise distributions 
and aligning the classification layers in a post-hoc fashion. 
Across a variety of scenarios, our proposal provides substantial improvements for CLPM 
(e.g., up to 49.76%, 50.05%, 44.69% and 40.16% on Split CIFAR-100, Split ImageNet-R, Split CUB-200 and Split Cars-196, respectively), 
and thus outperforms state-of-the-art approaches by a large margin. Based on such a strong baseline, 
critical factors and promising directions are analyzed in-depth to facilitate subsequent research.
