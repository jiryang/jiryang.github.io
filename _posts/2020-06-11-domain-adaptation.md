---
layout: post
title:  "Domain Adaptation: Learning to Learn - part 2"
date:   2020-06-11 00:00:01
categories: GeneralML
tags: domain_adaptation transfer_learning
excerpt: 도메인 적응 리뷰 및 예시
mathjax: true
---


지난 [포스트](https://jiryang.github.io/2020/06/08/domain-adaptation/)에 이어 시작합니다.

_Discrepancy-based: Architectural_ <br>
또다른 discrepancy-based 접근법 중에는 모델 네트워크의 architecture를 이용한 방법이 있습니다. 앞서 설명한 statistical 방식에서 $$\mathcal{X^S}$$와 $$\mathcal{X^T}$$를 하나의 네트워크에 넣어서 common feature representation을 학습했던 것과 달리, 두 domain 간의 유사하지만 다른 (사이비인가요..) 점들을 반영하기 위해 two-stream으로 네트워크를 구성하고, 직접적인 weight sharing 대신 weight regularizer function을 사용하여 layer 별 weight의 relation을 갖게 한 방식입니다. 아래 그림은 이러한 two-stream network를 사용한 DA의 대표적인 예입니다. Labeled $$\mathcal{X^S}$$ (및 labeled $$\mathcal{X^T}$$, if available)에 대해 softmax classification loss를 학습하고, 두 domain의 discrepancy를 줄이는 loss를 학습하고, 말씀드린 layer-wise regularization loss를 통해 각 layer의 weight 값에 difference bound( )hard 또는 soft)를 두는 식입니다.<br>
또다른 architectural method로는 class label 관련 정보는 weight matrix에, domain 관련 정보는 batch norm (BN)의 statistics에 저장된다는 점에 착안해 BN에서 $$\mathcal{D^T}$$의 mean과 std를 조정하도록 parameter를 학습시켜 domain discrepancy를 줄이는 Adaptive Batch Normalization (AdaBN) 이라는 방법도 있습니다.<br>
또한, internal layer의 neuron 중 일부는 다양한 domain의 input에 activate되는 것도 있는 반면 일부는 특정 domain에 specific하게 activate 되는 것들이 있다는 점에 착안하여, 하나의 네트워크에 $$\mathcal{X^S}$$와 $$\mathcal{X^T}$$를 모두 입력하면서 domain-specific한 neuron의 activation을 zero로 masking 하면서 domain-general한 feature representation을 더욱 '잘' 학습하도록 하는 domain-guided dropout과 같은 방식도 architectural approach로 분류됩니다.


![Fig1](https://jiryang.github.io/img/related_weights.PNG "Two-Stream Architecture with Related Weights"){: width="80%"}{: .aligncenter}


_Discrepancy-based: Geometric_ <br>
$$\mathcal{D^S}$$와 $$\mathcal{D^T}$$의 차이 (domain shift)를 줄여주는 (두 domain 간의 correlation이 높은) 제3의 manifold로 $$\mathcal{X^S}$$와 $$\mathcal{X^T}$$를 projection시킨 후 domain-invariant feature representation을 학습하게 하는 방식이 geometric한 discrepancy-based approach에 속합니다. Geometric 방식 중에 $$\mathcal{D^S}$$와 $$\mathcal{D^T}$$ 사이의 interpolation path 상에 새로운 dataset들을 조합해 만들어내고, 이것들로부터 domain-invariant한 feature를 뽑아내어 classification에 이용하는 Deep Learning for domain adaptation by Interpolating between Domains (DLID)라는 다소 작위적인 이름을 가진 모델이 있긴 한데, pre-processing이 엄청난 것으로 보여서 설명 생략하고 넘어가겠습니다. 필요하신 분은 [링크](http://deeplearning.net/wp-content/uploads/2013/03/dlid.pdf) 참조하세요.


**Adversarial-based**<br>
Generator-discriminator의 minimax game으로 합성 데이터를 만들어내는 GAN이 큰 성공을 거두면서, GAN에서 착안한 DA 방법론들도 등장하게 되었습니다. 아래는 adversarial-based DA를 일반화한 그림입니다. 우선 labeled $$\mathcal{X^S}$$로 classifier를 학습시킵니다. 그 다음 GAN_{source)을 사용해서 synthesized source data를 만들고, 앞의 classifier로 synthesized source data의 class label을 구합니다. 이 GAN과 parallel한  전체적인 구조는 유지한 상태에서 구현의 디테일을 어떻게 가져가느냐, 회색 블럭의 옵션들을 어떻게 선택하느냐에 따라 모델이 달라진다고 할 수 있겠습니다. 특히 첫번째 회색 블럭의 선택지에 따라 합성 데이터를 실제로 만들어내는 부분이 포함된 generative 접근법과, discriminator의 동작 방식을 본뜬 domain discriminator를 '반대로' 학습시켜 domain confusion을 일으키도록 해서 모델을 domain-invariant하게 만드는 non-generative 접근법으로 구분할 수 있습니다.<br>

![Fig2](https://jiryang.github.io/img/adversarial_DA.PNG "Generalized Architecture of Adversarial DA"){: width="80%"}{: .aligncenter}


_Adversarial-based: Generative_<br>
Coupled GAN(CoGAN)에서는 $$\mathcal{X^S}$$와 $$\mathcal{X^T}$$를 합성하는 두 개의 GAN을 parallel하게 놓고, low-level layer들의 weight를 공유시킴으로써 

![Fig3](https://jiryang.github.io/img/cogan.PNG "Coupled Generative Adversarial Networks"){: width="80%"}{: .aligncenter}


_Adversarial-based: Non-generative_<br>
asdfasdfdf


**Reconstruction-based**<br>
Source나 target domain 데이터를 reconstruct하여 intra-domain specificity와 inter-domain indistinguishability를 높이는 방식<br>

_Reconstruction-based: Encoder-decoder_<br>
asdfasdfdf


_Reconstruction-based: Adversarial_<br>
asdfasdfdf