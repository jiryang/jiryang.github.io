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


이와 같은 RL의 Agent - Environment interaction은 Markov Decision Process (MDP)로 표현될 수 있습니다 (MDP에 대해 궁금하시면 [여기](https://towardsdatascience.com/introduction-to-reinforcement-learning-markov-decision-process-44c533ebf8da) 참고). MDP는 유한한 environment state space인 $$S$$, 각 state에서 가능한 action의 집합인 $$A$$ (state $$s$$에서 가능한 action들의 집합은 $$A_s$$로 표현), next state s'가 current state s와 action a에 의해 결정된다는 state transition function $$P_a(s, s')$$ (=$$Pr(s_{t+1} = s' \vert s_t = s, a_t = a)$$), 그리고 action $$a$$에 의해 state $$s$$가 $$s'$$로 전이될 때 주어지는 reward $$R_a(s, s')$$의 tuple ($$S$$, $$A$$, $$P_a$$, $$R_a$$)으로 나타낼 수 있고, 여기에 immediate vs future reward 사이의 중요도를 구분함으로써 continuous task에서 reward가 무한대에 이르는 것을 막아주기 위한 discount factor $$\gamma$$를 추가한 ($$S$$, $$A$$, $$P_a$$, $$R_a$$, $$\gamma$$)의 5-tuple로 표현합니다.


