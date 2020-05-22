---
layout: post
title:  "FaceID-GAN: Learning a Symmetry Three-Player GAN for Identity-Preserving Face Synthesis"
date:   2020-05-22 10:59:59
categories: Deepfake
tags: deepfake faceidgan id-preserving
excerpt: FaceID-GAN 논문 리뷰
mathjax: true
---

지난 포스트에서 [FSGAN](https://jiryang.github.io/2020/05/14/FSGAN-review/)에 대해 살펴보았었습니다.

'Deepfake'에 사용되는 face-swap이 source 인물의 얼굴을 target 영상에 잘라붙이는 방식으로 동작한다는건 많이들 아실텐데요, 간단히 설명하고 넘어가겠습니다. 'Deepfake'에는 1.5개의 AutoEncoder가 필요합니다 (Encoder 부분은 share되고, decoder만 별도로 학습하면 되니깐 1.5개라고...). Single encoder에 a set of source 및 a set of target 얼굴 이미지를 입력하고 차원을 축소하여 latent face를 만든 다음, source/target 별도의 decoder로 입력 이미지를 복원하도록 학습을 시킵니다 (결과물이 동영상인 경우라면 target 이미지들은 동영상을 ffmpeg 같은거로 frame으로 뜯어내어서 얼굴들을 가져오면 될꺼고요, source 쪽도 마찬가지로 영상에서 가져온거든 별개의 이미지들을 가져다넣은 것이든 상관없지만, 가능하면 target에서 나타나는 표정과 얼굴 각도의 variation을 포함하는 super-set을 넣어주면 합성 quality가 좀 더 좋습니다). 

이제 encoder는 source와 target얼굴 모두의 (공통의) latent face를 만들 수 있게 되었고 decoder는 여기서 각각의 원래 얼굴을 복원할 수 있게 되었기 때문에, target decoder를 떼어내고 학습된 encoder - source decoder로만 조합을 해서 target 얼굴 이미지를 입력하면 이 attribute(표정, 각도 등)를 따르는 source의 얼굴을 합성하게 됩니다. 이 합성된 얼굴을 잘라서 원래의 target 이미지에 티 안나게 잘 blending시키면 deepfake가 완성됩니다.

Source랑 target이 좀 헷갈릴 수 있는데, 잘 읽어보시면 이해가 될꺼예요. Face A가 target이고 Face B가 source겠죠?
![Fig1](https://jiryang.github.io/img/faceswap_autoencoder.png "How Deepfake Works"){: .center}


이 방식은 비교적 간단한 네트워크를 사용하고, 데이터도 그렇게 많이 들어가지 않으며, source-target 간의 1-to-1 학습이기 때문에 학습이 쉽고 합성 quality도 좋은 편입니다. 하지만 source나 target 인물이 바뀔때마다 학습을 새로 해야한다는 점, source의 데이터가 많아야 한다는 점이 이 모델의 약점입니다.

이후 얼굴 합성 연구는 pose-guided 방식의 face reenactment 알고리즘들이 메인스트림인 듯 합니다. 지금까지 전체적으로 봤을 때,  

(Deepfake에 대해 궁금하시다면 이런 [reference](http://news.seoulbar.or.kr/news/articleView.html?idxno=1817)도 있습니다 :) )