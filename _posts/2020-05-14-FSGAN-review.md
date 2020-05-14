---
layout: post
title:  "FSGAN: Subject Agnostic Face Swapping and Reenactment"
date:   2020-05-14 23:59:59
categories: Deepfake
tags: deepfake fsgan faceswap 
excerpt: FSGAN 논문 리뷰
mathjax: true
---

오늘은 작년 코엑스에서 열렸던 ICCV'19에서 소개된 `face swap` 알고리즘인 FSGAN에 대해 소개해보겠습니다.

헷갈리시는 분들을 위해 `face swap`과 `face reenactment`의 차이를 그림으로 보여주면서 SIGGRAPH 스타일로 논문을 시작합니다.
![Fig1](https://jiryang.github.io/img/faceswap_vs_facereenactment.JPG "Face Swap vs. Face Reenactment")

이 논문은 왼쪽의 `face swap`에 관한 내용입니다.

Deepfake에 악용되었을 경우 `face reenactment`가 더 파장이 클 수 있겠으나 아직은 dummy actor를 놓고 swapping 하는 방식이 quality나 throughput이 더 낫다고 생각합니다. 하지만 양쪽 모두 기술이 발전하고 있으니 계속 지켜봐야죠.

