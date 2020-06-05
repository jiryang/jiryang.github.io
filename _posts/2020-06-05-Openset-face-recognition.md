---
layout: post
title:  "Open-Set Face Recognition: SphereFace, CosFace, ArcFace, then CurricularFace"
date:   2020-06-05 00:00:01
categories: Deepfake
tags: openset_face_recognition sphereface cosface arcface curricularface curriculum_learning
excerpt: 얼굴인식에서의 커리큘럼 학습
mathjax: true
---

오늘은 triplet을 이용한 FaceNet 이후의 xxxFace 시리즈에 대해 이야기를 해보려고 합니다.<br>
FaceNet에 대한 지난[포스트](https://jiryang.github.io/2020/05/23/FaceNet-and-one-shot-learning/)에서 open-set face recognition에 대한 문제를 이야기했습니다. 일반적인 classification의 경우 각 class에 assign된 data로 모델을 학습하고, unseen test data를 어느 class에 할당할 것인지를 결정합니다. 이런걸 closed-set 문제라고 합니다. 하지만 지난 포스트에서 언급했던 face recognition과 같은 경우 학습되지 않은 얼굴을 어떻게 인식할 것인지의 문제가 생깁니다. 할당할 class가 없기 때문에 FaceNet에서는 probe face를 gallery에 있는 얼굴들과 비교하기 위한 low-dimensional embedding을 triplet을 구성하여 학습시켰던 것을 기억하실 것입니다. 이런걸 open-set 문제라고 하는데요, 결국 open-set face recognition은 'real domain에서 다른 얼굴이라면 feature space에서도 다른 얼굴이라고 인식하는'것을 학습하는 것이고, 다르게 표현한다면 face image를 feature space로 mapping하기 위한 metric을 학습하는 문제라고 할 수 있습니다. 이 feature space는 서로 다른 얼굴들에 대한 구분력을 최대화해야 할 것이므로 discriminative margin을 최대화하는 방향으로의 mapping metric을 배워야 합니다.

![Fig1](https://jiryang.github.io/img/closedset_vs_openset.PNG "Closed vs Open-Set Face Recognition"){: width="70%"}{: .aligncenter}


Discriminative feature를 효과적으로 배워보자는 두 가지 시도가 우선 있었습니다. Contrasive loss(또는 pairwise ranking loss)는 anchor-positive, anchor-negative pair를 구성해서 각 이미지를 Siamese network에 집어넣어 나온 feature들을 이용하여 다음의 loss를 최적화하게 됩니다:<br>
$$L_{contrasive} = (1-Y) \frac 1 2 (\Vert f(x^i) - f(x^j) \Vert)^2 + Y \frac 1 2{max(0, m - \Vert f(x^i) - f(x^j) \Vert)}^2$$
FaceNet은 


FaceNet이 triplet을 만들어서 easy-to-hard 순서로 학습한다는 부분에서 curriculum learning에 대해서도 잠깐 언급했었지요.


얼굴 인식 모델을 학습하여 


논문링크: 
[SphereFace](https://arxiv.org/pdf/1704.08063.pdf)
[CosFace](https://arxiv.org/pdf/1801.09414.pdf)
[ArcFace](https://arxiv.org/pdf/1801.07698.pdf)
[CurricularFace](https://arxiv.org/pdf/2004.00288.pdf)