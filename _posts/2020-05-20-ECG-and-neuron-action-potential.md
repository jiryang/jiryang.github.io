---
layout: post
title:  "ECG, Neuron's Action Potential, and ANN"
date:   2020-05-20 10:00:00
categories: Health
tags: ecg neuron actionpotential ann neuralnet
excerpt: ECG와 뉴럴 네트워크의 관계?
mathjax: true
---

오늘은 ECG 파형에 대해 간단히 이해를 해보고 (ECG 데이터를 가지고 리서치를 하려면 배경 지식이 좀 있으면 좋겠죠), 이 이야기를 뉴럴 네트워크까지 이어가보도록 하곘습니다.

심근세포의 action potential은 polarization-depolarization에 의해 발생합니다. 
뉴런의 action potential과 굉장히 비슷하기 때문에 뉴런의 예로 설명해보겠습니다.

지질로 구성된 생물의 세포막은 안쪽과 바깥쪽 voltage의 차이를 유지합니다 (막 안쪽이 -70mV 정도로 negative임). 어떤 세포들은 이 막전위를 constant하게 유지하지만 일부는 voltage가 변하기도 하며, 특히 신경세포와 근세포들은 매우 빠르고 연속적으로 이러한 voltage 변화를 일으켜서 세포간에 신호를 전달하는 기능을 수행합니다.

![Fig1](https://jiryang.github.io/img/action_potential.png "Neuron's Action Potential"){: width="50%"}{: .center}

