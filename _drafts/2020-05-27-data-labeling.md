---
layout: post
title:  "Efficient Data Labeling: Online Learning & Active Learning"
date:   2020-05-27 02:00:00
categories: GeneralML
tags: bigdata labeling online_learning active_learning
excerpt: 머신러닝 모델을 학습시키기 위해 어떤 데이터를 얼마나, 어떻게 레이블링 해야 하는가 (예고)
mathjax: true
---

한 업체에 다녀왔습니다. 이곳도 'AI는 해야겠는데 어떻게 할 지는 모르는' 곳 중 하나였습니다.
많은 기업들이 비용을 들이자니 palpable한 효과가 잘 그려지지 않고, 손을 놓고 있자니 뒤처질 것 같아 불안한 상태인 것 같습니다.
그래서 동종 업계의 success나 failure case를 요구하기도 하고, PoC를 해보자고 하는 것이겠지요.


"AI를 언제든 적용할 수 있도록 데이터는 미리 모아두었습니다."
PoC를 하자는 업체에서 많은 데이터를 받았습니다. 그런데 원하는 모델에 넣기에 레이블이 제대로 되어있지 않습니다. PoC 기간과 예산은 충분하지 않습니다. 어떻게 하면 될까요?


많은 양의 quality data가 필요하다는 점은 현대 supervised deep learning의 커다란 단점으로 지적되고 있습니다.
(AAAI keynote speech에서 얀 르쿤 박사께서도 지적하신 부분이지요 [Self-supervised learning: The plan to make deep learning data-efficient](https://bdtechtalks.com/2020/03/23/yann-lecun-self-supervised-learning/amp/))

... (미완성)