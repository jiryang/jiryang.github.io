---
layout: post
title:  "FSGAN: Subject Agnostic Face Swapping and Reenactment"
date:   2020-05-14 23:59:59
categories: Deepfake
tags: deepfake, fsgan, faceswap 
excerpt: FSGAN 논문 리뷰
mathjax: true
---

오늘은 작년 코엑스에서 열렸던 ICCV'19에서 소개된 `face swap` 알고리즘인 FSGAN에 대해 소개해보겠습니다.

헷갈리시는 분들을 위해 `face swap`과 `face reenactment`의 차이를 그림으로 보여주면서 SIGGRAPH 스타일로 논문을 시작합니다.
![](https://jiryang.github.io/img/faceswap_vs_facereenactment.JPG)