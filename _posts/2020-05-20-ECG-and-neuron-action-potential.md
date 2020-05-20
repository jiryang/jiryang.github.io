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

지질로 구성된 생물의 세포막은 안쪽과 바깥쪽 voltage의 차이를 유지합니다 (막 안쪽이 -70mV 정도로 negative임). 어떤 세포들은 이 막전위를 constant하게 유지하지만 일부는 voltage가 변하기도 하며, 특히 신경세포와 근세포들은 매우 빠르고 연속적으로 이러한 voltage 변화를 일으켜서 세포내에서 신호를 전달하는 기능(dendrite-to-axon terminal)을 수행합니다.

![Fig1](https://jiryang.github.io/img/action_potential.png "Neuron's Action Potential"){: width="50%"}{: .center}


뉴런의 axon은 myelin이라는 지방으로 코팅이 되어있는데요, myelin의 역할은 위의 action potential이 전달되는 속도를 배가시켜줍니다. 단위시간에 더 많은 impulse를 전달하게 되어 결국은 더 '강한' 신호가 전달되도록 하는 효과가 납니다.
![Fig2](https://jiryang.github.io/img/neuron.png "Neuron"){: width="50%"}{: .center}


뇌에서 A-to-B의 전달 신호의 세기는 두 지점을 연결하는 뉴런의 갯수, 그 연결을 구성하는 뉴런들의 synapse의 수, 구성 뉴런들 사이의 neurotransmitter의 양 뿐만 아니라 저 myelin 또한 영향을 미칩니다. 이를 굉장히 단순화하여 모델링 한 것이 ANN이고, 여기선 신호 전달의 세기가 weight의 magnitude로 표현이 되지요.

![Fig3](https://jiryang.github.io/img/ann1.jpg "Biological vs. Artificial Neuron"){: width="50%"}{: .center}

이제 biological neuron과 ANN의 neuron 사이의 analogy가 좀 더 명확해졌나요?