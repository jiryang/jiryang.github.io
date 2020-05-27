---
layout: post
title:  "Efficient Data Labeling: Online Learning & Active Learning"
date:   2020-05-27 12:00:00
categories: GeneralML
tags: bigdata labeling online_learning active_learning
excerpt: 머신러닝 모델을 학습시키기 위해 어떤 데이터를 얼마나 레이블링 해야 하나요?
mathjax: true
---

어제 한 업체에 다녀왔습니다. 이곳도 'AI는 해야겠는데 어떻게 할 지는 모르는' 곳 중 하나였습니다.
많은 기업들이 비용을 들이자니 palpable한 효과가 잘 그려지지 않고, 손을 놓고 있자니 뒤처질 것 같아 불안한 상태인 것 같습니다.
그래서 동종 업계의 success나 failure case를 요구하기도 하고, PoC를 해보자고 하는 것이겠지요.

PoC는 좋지요. 적은 규모의 비용으로 솔루션을 검증한 후 결정하겠다는건 합리적인 생각입니다. 그런데 데이터는 있으신가요?


레이블이 제대로 되어있지 않은 데이터를 받았습니다. 양은 엄청나게 많습니다. PoC 기간과 예산은 충분하지 않습니다. 어떻게 하면 될까요?





*논문을 읽다가 '이걸 적어야겠다'는 생각이 들어서 material을 정리하다보면 그 전에 이것도 하나 적으면 좋겠고, 그 옆에 저것도 하나 적으면 좋겠고... 머신러닝의 각 화두들이 어디서 뚝 떨어진 것이 아니라 히스토리가 있고 연결 관계들이 있는 것들이어서... 손은 딸리는데 큰일입니다.*