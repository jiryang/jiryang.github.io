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


RL에서 풀고자 하는 문제는 이렇게 Environment 안에서 sensing 및 acting을 하는 agent가 어떻게 하면 goal을 달성하기 위한 optimal한 action을 선택할 지를 학습하는 것입니다. 다시 말하면 goal을 달성하기까지 Environment로부터 받는 reward를 maximize하기 위한 action 선택 전략, 즉 $$optimal \; policy$$를 학습하는 것이라고 할 수 있겠습니다. 그런데 이 reward (혹은 penalty)는 처음부터 모든 state에서 주어지는 것이 아닙니다. 체스와 같은 보드게임의 예를 들면 최종적으로 게임을 이겼는지의 여부에 따라 reward가 주어지는 것이지, 그 과정까지의 action에 대해서는 problem domain에 대한 어지간한 지식이 있지 않고서는 reward/penalty 값을 assign하기가 쉽지 않습니다. 현재의 action이 게임의 승패에 얼마나 영향을 미쳤는지를 거꾸로 추적하여 reward/penaly를 매겨야 하기 때문에 이러한 RL은 temporal credit assignment problem이라고도 볼 수 있습니다. Supervised classification과 빗대어 보자면 classification error gradient를 backpropagation을 통해 각 edge에 spatial하게 assign하여 weight update를 했던 것과 어떤 측면으로는 좀 비슷하게, RL은 이전의 action selection $$policy$$들의 '기여도'를 추정하여 temporal하게 각 policy의 reward를 update하는 것이죠.


$$Optimal \; policy$$를 찾기 위해 RL problem을 Markov Decision Process (MDP)로 formalize할 수 있습니다 (MDP에 대해 궁금하시면 [여기](https://towardsdatascience.com/introduction-to-reinforcement-learning-markov-decision-process-44c533ebf8da) 참고). MDP는 유한한 environment state space인 $$S$$, 각 state에서 가능한 action의 집합인 $$A$$ (state $$s$$에서 가능한 action들의 집합은 $$A_s$$로 표현), next state $$s_{t+1}$$이 current state $$s_t$$와, 당시의 action $$a_t$$에 의해 결정된다는 state transition function $$s_{t+1}=\delta(s_t, a_t)$$, 그리고 action $$a_t$$에 의해 state $$s_t$$가 $$s_{t+1}$$로 전이될 때 주어지는 reward function $$r(s_t, a_t)$$의 tuple <$$S$$, $$A$$, $$\delta$$, $$r$$>으로 나타낼 수 있고, 여기에 immediate vs delayed (future) reward 사이의 중요도를 구분함으로써 continuous task에서 reward가 무한대에 이르는 것을 막아주기 위한 discount factor $$\gamma$$를 추가한 <$$S$$, $$A$$, $$\delta$$, $$r$$, $$\gamma$$>의 5-tuple로 표현합니다.


Current state $$s_t$$로부터 다음 action $$a_t$$를 찾는 $$policy$$ ($$\pi(s_t)=a_t$$)를 $$\pi : S \rightarrow A$$로 정의합니다. 그리고 이 $$\pi$$에 의해 얻어지는 cumulative reward를 $$V^{\pi}(s_t)$$라고 하면 다음과 같은 정의가 성립합니다:<br>
$\qquad$ $$V^{\pi}(s_t) \equiv r_t + \gamma r_{t+1} + \gamma^2r_{t+2} + ... \equiv \sum^{\infty}_{i=0}\gamma^ir_{t+i}$$<br><br>
이제 $$optimal \; policy \pi^{\ast}$$를 다음과 같이 정의할 수 있습니다:<br>
$\qquad$ $$\pi^{\ast} \equiv argmax_{\pi} V^{\pi}(s), (\forall s)$$<br><br>
위 정의는 '어떤 state에서 $$optimal \; policy$$란 해당 state로부터의 cumulative reward를 최대화하는 $$policy$$이다'라는 의미로 직관적입니다.


**Q Learning**
위의 정의에 따라 given state $$s$$에서의 $$optimal \; policy$$를 다음과 같이 표현할 수 있습니다:<br>
$\qquad$ $$\pi^{\ast}(s) = argmax_{a} \left[r(s, a) + \gamma V^{\ast}(\delta(s, a)) \right]$$<br><br>