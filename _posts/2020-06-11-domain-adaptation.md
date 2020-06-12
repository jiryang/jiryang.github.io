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
Generator-discriminator의 minimax game으로 합성 데이터를 만들어내는 GAN이 큰 성공을 거두면서, GAN에서 착안한 DA 방법론들도 등장하게 되었습니다. 우선 labeled $$\mathcal{X^S}$$로 classifier를 학습시킵니다. 그 다음 GAN<sub>source</sub>를 사용해서 synthesized source data를 만들고, 앞의 classifier로 synthesized source data의 class label을 구합니다. GAN<sub>source</sub>와 parallel한 GAN<sub>target</sub>을 domain-invariant하게 만들어 놓고 동시에 synthesized target data를 생성하게 하면, 이 synthesized target data는 앞의 synthesized source data와 동일한 label을 가지면서도 $$\mathcal{X^T}$$의 분포를 따르는 ($$\mathcal{X^T}$$처럼 생긴) output이 나올꺼라는 것이 기본 아이디어입니다. 그 결과 labeled synthesized target data가 만들어지는 셈이기 때문에, 이것들로 classifier를 학습하면 target classifier가 만들어지는 것이죠. 좀 헷갈릴 수도 있는데 잘 생각해보면 말이 됩니다. 아래는 adversarial-based DA를 일반화한 그림입니다. 전체적인 구조는 유지한 상태에서 구현의 디테일을 어떻게 가져가느냐, 회색 블럭의 옵션들을 어떻게 선택하느냐에 따라 모델이 달라진다고 할 수 있겠습니다. 특히 첫번째 회색 블럭의 선택지에 따라 합성 데이터를 실제로 만들어내는 부분이 포함된 generative 접근법과, discriminator의 동작 방식을 본뜬 domain discriminator를 '반대로' 학습시켜 domain confusion을 일으키도록 해서 모델을 domain-invariant하게 만드는 non-generative 접근법으로 구분할 수 있습니다.<br>

![Fig2](https://jiryang.github.io/img/adversarial_DA.PNG "Generalized Architecture of Adversarial DA"){: width="80%"}{: .aligncenter}


_Adversarial-based: Generative_<br>
Coupled GAN(CoGAN)에서는 $$\mathcal{X^S}$$와 $$\mathcal{X^T}$$를 합성하는 두 개의 paralle한 GAN의 low-level layer들의 weight를 공유시킴으로써 higher (혹은 deeper) layer들이 두 domain을 다 cover하는 (domain-invariant한) 특성을 학습하게끔 하는 방식입니다.

![Fig3](https://jiryang.github.io/img/cogan.PNG "Coupled Generative Adversarial Networks"){: width="80%"}{: .aligncenter}


[Pixel-level domain transfer network](https://arxiv.org/pdf/1603.07442.pdf)는 (1) Encoder-Decoder로 구성된 domain converter; (2) real-fake discriminator; (3) domain-discriminator의 3개 네트워크로 구성되어 각각이 다음과 같은 역할을 수행합니다 (여기서는 어느 정도의 labeled $$\mathcal{X^T}$$가 필요합니다):<br><br>
(1) Source data를 입력받아 target data를 합성<br>
(2) Real 또는 synthesized target data를 입력받아 real/fake binary classification 수행<br>
(3) (1)의 source와 동일한 class의 labeled target data pair를 입력받아 두 data의 association이 있는지 없는지 binary classification을 수행<br><br>
(2)의 역할 덕분에 (1)의 네트워크는 _realistic fake target data_ 를 만들게 되고, (3)의 역할 덕분에 _realistic fake target data associated to the source_ 를 만들게 되는 것입니다.

![Fig4](https://jiryang.github.io/img/pixel_level_domain_transfer.PNG "Architecture of Pixel-Level Domain Transfer"){: width="80%"}{: .aligncenter}

 
하나만 더 예를 들어보죠. [Unsupervised Pixel-Level Domain Adaptation with Generative Adversarial Networks](https://arxiv.org/pdf/1612.05424.pdf) 에서는 labeled data가 충분치 않아 rendering한 (label은 자동으로 붙겠죠?) source data로 모델을 학습시켜서 real source data에도 general하게 사용하려는 목적으로 DA를 수행합니다. Synthetic이 두 종류가 들어가서 헷갈리는데요, 여기선 rendered가 $$\mathcal{X^S}$$이고, unlabeled real이 $$\mathcal{X^T}$$인 셈입니다. 아래 그림에서 보시면 (여긴 rendered를 synthetic이라 표현) generator G는 noise vector와 rendered source data를 입력으로 받아서 fake source data를 만듭니다. Task-specifc classifier T는 rendered source data와 fake source data를 입력받아 known class label로 분류가 되도록 학습을 하고, discriminator D는 real source data와 fake source data를 입력받아 real/fake를 구분하는 역할을 함으로써 fake를 real에 가깝게 만들어줍니다. 3개 네트워크를 함께 학습시키고 나면 rendered data를 입력받아; real source와 가까우면서도; rendered data의 class로 분류되는 fake data를 생성하게 되는거죠.

![Fig5](https://jiryang.github.io/img/unsupervised_pixel_level_DA.PNG "Architecture of Unsupervised Pixel-Level DA with GAN"){: width="60%"}{: .aligncenter}


_Adversarial-based: Non-generative_<br>
Adversarial-based non-generative 방식의 DA는 adversarial training의 discriminator의 아이디어를 차용합니다. 예시 모델을 보면서 설명하도록 하지요. 아래 Domain-Adversarial Neural Network([DANN](https://arxiv.org/pdf/1505.07818.pdf)) 네트워크를 보시면 앞단의 feature extractor (green)는 일반적인 single flow CNN과 별다르지 않게 되어있으나, 그 뒤에는 path가 label predictor (blue)와 domain classifier (red) 둘로 갈립니다. 이 네트워크에는 $$\mathcal{X^S}$$와 $$\mathcal{X^T}$$가 모두 입력되는데요, 입력 데이터의 domain에 따라 green, blue, red 모듈이 선택적으로 동작하게 됩니다. 우선 일반적인 classifier인 green-blue 조합은 $$\mathcal{X^S}$$일 때만 동작하여 class label을 분류하도록 학습하게 됩니다. Domain classifier인 green-red 조합은 $$\mathcal{X^S}$$와 $$\mathcal{X^T}$$의 경우 모두 동작하여 해당 input이 $$\mathcal{D^S}$$에서 왔는지 $$\mathcal{D^T}$$에서 왔는지를 구분하도록 학습합니다. 이 domain classifier의 학습에 DANN의 특징이 있는데요, red 모듈은 일반적인 backpropagation을 사용하여 학습하는 반면, domain classifier loss가 feature extractor 모듈로 backpropagate 될 때 gradient reversal layer를 거쳐서 gradient의 부호가 바뀌게 되고, 이 역전된 값으로 나머지 feature extractor 모듈의 학습이 이루어집니다. 이는 feature extractor로 하여금 domain을 잘 구분하지 못하도록 학습하게 하여서 결국 feature extractor의 결과가 domain-invariant해 지는 효과를 낳습니다. 학습이 완료되면 feature extractor는 label predictor의 loss로 인해 class 구분이 되는 feature를, domain classifier의 reversed loss로 인해 domain이랑 상관없이 class 구분이 되는 feature를 만들게 되는거죠.

![Fig6](https://jiryang.github.io/img/DANN.PNG "Domain-Adversarial Neural Network"){: width="80%"}{: .aligncenter}


또다른 non-generative approach의 예로 Adversarial Discriminative Domain Adaptation ([ADDA](https://arxiv.org/pdf/1702.05464.pdf))가 있습니다. ADDA는 DANN과는 달리 source와 target 별도의 feature extractor를 가지고 있습니다. 아래 그림에서와 같이 일단 $$\mathcal{X^S}$$로 classifier pre-train을 하고나서, adversarial adaptaion을 위해 학습된 classifier의 feature extractor 부분을 떼서 domain discriminator에 붙입니다. Target feature extractor는 learned source feature extractor의 weight로 initialize 시키고, adversarial adaption 과정에서 추가적으로 학습됩니다. Source 쪽 feature extractor는 더이상 학습하지 않고 weight가 고정됩니다 (그림에서 점선으로 표기되면 weight 고정이란 의미). Target feature extractor를 학습하는 domain discriminator의 adversarial adaptation은 source와 target 여부를 invert시켜서 학습하여 앞선 DANN에서의 gradient reversal layer를 적용한 것과 비슷한 효과를 내어, 학습된 target feature가 source와 domain 구분은 잘 안되면서 classification은 유사하게 나오는 특성을 가지게 된다는 아이디어입니다. 이렇게 학습된 target feature extractor를 앞서 학습한 source classifier에 붙여서 test를 하면 unlabeled $$\mathcal{X^T}$$을 괜찮게 classify하는 결과가 나오는거죠.

![Fig7](https://jiryang.github.io/img/ADDA.PNG "Adversarial Discriminative Domain Adaptation"){: width="80%"}{: .aligncenter}


**Reconstruction-based**<br>
Source나 target domain 데이터를 reconstruct하여 intra-domain specificity와 inter-domain indistinguishability를 높이는 방식<br>

_Reconstruction-based: Encoder-decoder_<br>
asdfasdfdf


_Reconstruction-based: Adversarial_<br>
asdfasdfdf