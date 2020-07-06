---
layout: post
title:  "Deep Neuroevolution - part3"
date:   2020-07-06 00:00:01
categories: ReinforcementLearning
tags: reinforcement_learning q_learning policy_gradient
excerpt: Traditional Reinforcement Learning
mathjax: true
---

오랜만에 쓰네요.

엎어진 김에 쉬어간다고 traditional reinforcement learning (RL)에 대해 정리하고 넘어가도록 하겠습니다.

RL에 대해 조금이라도 찾아보신 분들이라면 Agent가 Environment로 Action을 가하고, Environment로부터 State와 Reward를 받아오는 아래 그림이 매우 익숙하실 것입니다. 굳이 한 줄 쓰자면, Environment로부터 time $$t$$의 current state $$S_t$$를 받은 Agent가 이 state에서 가능한 action $$A_t$$를 수행하면 이 action에 따른 reward $$R_{t+1}$$을 받습니다. 이후 Environment에서는 state를 $$S_{t+1}$$로 업데이트하고... 이 과정을 goal state가 발견될 때까지 반복하는 것입니다. 

$$S_0, A_0, R_1, S_1, A_1, R_2, S_2, A_2, R_3, ...$$

![Fig1](https://jiryang.github.io/img/reinforcement_learning.jpg "Reinforcement Learning"){: width="80%"}{: .aligncenter}

