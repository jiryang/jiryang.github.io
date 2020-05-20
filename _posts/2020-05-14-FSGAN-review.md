---
layout: post
title:  "FSGAN: Subject Agnostic Face Swapping and Reenactment"
date:   2020-05-14 13:59:59
categories: Deepfake
tags: deepfake fsgan faceswap 
excerpt: FSGAN 논문 리뷰
mathjax: true
---

오늘은 작년 코엑스에서 열렸던 ICCV'19에서 소개된 `face swap` 알고리즘인 FSGAN에 대해 소개해보겠습니다.


헷갈리시는 분들을 위해 `face swap`과 `face reenactment`의 차이를 그림으로 보여주면서 SIGGRAPH 스타일로 논문을 시작합니다.

![Fig1](https://jiryang.github.io/img/faceswap_vs_facereenactment.JPG "Face Swap vs. Face Reenactment")


이 논문은 왼쪽의 `face swap`에 관한 내용입니다.
(Deepfake에 악용되었을 경우 `face reenactment`가 더 파장이 클 수 있겠으나 아직은 dummy actor를 놓고 swapping 하는 방식이 quality나 throughput 측면에서 더 낫습니다. 하지만 양쪽 모두 기술이 발전하고 있으니 계속 지켜봐야죠.)


Training data의 분포를 따르는 새로운 instance를 합성하는 `GAN (Generative Adversarial Network)`이 발명되고 수많은 분야에 적용 및 개선이 되어왔습니다. 이후 one-hot vector로 가이드를 줘서 원하는 방향으로 합성 결과를 뽑아내는 [cGAN](https://arxiv.org/pdf/1411.1784.pdf) 방식이 고안되었으며, 이어서 conditional vector의 dimension을 확장하여 한 이미지로 다른 이미지의 스타일을 가이드하여 변경/합성시키는 [pix2pix style transfer](https://arxiv.org/pdf/1611.07004.pdf) 방식이 개발되었습니다. 여기까지가 'innovation' 이라고 하면, 이 이후로는 성능을 최적화한다거나 scale을 높인다거나, 특정 도메인에 특화한다거나 하는 수많은 minor improvement 연구 결과물들이 쏟아져 나오게 되었죠.

| ![Fig2](https://jiryang.github.io/img/tech_s_curve.png "Innovation S-Curve"){: width="50%"}{: .center} |
|:--:|
|*(연구도, 진화도, 비지니스도 innovation S curve를 따르는 것 같습니다)*|


최근 FSGAN과 같이 ID 얼굴사진(1)과 Attribute 얼굴사진(2)을 입력하여, (2)의 표정을 따라하는 (1)의 얼굴을 만들어내는 모델들이 많이 개발되고 있는데요, fewer-shot으로 하면서 ID preserving을 얼마나 잘 하는지가 이 분야의 가장 큰 과제인 것 같습니다. Demo 영상에서의 결과물이 썩 괜찮았던것 같아서 FSGAN에 많은 기대를 했었는데요, 안타깝게도 아직 ID preserving 성능이 썩 좋지는 않은 것 같습니다.

[![FSGAN Demo](https://jiryang.github.io/img/fsgan_demo.PNG)](https://www.youtube.com/watch?v=BsITEVX6hkE)




