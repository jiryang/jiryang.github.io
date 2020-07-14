---
layout: post
title:  "Deep Neuroevolution - part3"
date:   2020-07-14 00:00:01
categories: ReinforcementLearning
tags: reinforcement_learning q_learning
excerpt: Value-Based Reinforcement Learning
mathjax: true
---

오랜만에 쓰네요.

엎어진 김에 쉬어간다고 traditional reinforcement learning (RL)에 대해 정리하고 넘어가도록 하겠습니다. 
<!--Deep Q Networks (DQN)를 알기 위한 배경지식을 갖추려고 살펴보는 것이기 때문에 일단은 Q learning만 살펴보도록 하고, 다른 traditional RL 방법들인 합니다. Q learning에 대해 예제를 통해서 설명해 놓은 글들은 많기도 하고, Q learning을 쓰는 것보다는 이 알고리즘이 동작하는 원리에 대해 이해하는 것이 더 필요하기 때문에 이론적으로 알아보도록 하겠습니다.-->

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


RL task는 두 가지 종류로 나눌 수 있습니다. 하나는 학습을 통해 정확한 expected reward를 계산해내는 value-based method이고, 다른 하나는 취해야 하는 action을 학습하는 policy-based method 입니다. 각각의 경우를 대표적인 알고리즘으로 설명하겠습니다.

**Q-Learning (Value-based)**<br><br>
위의 정의에 따라 given state $$s$$에서의 $$optimal \; policy$$를 다음과 같이 표현할 수 있습니다:<br>
$\qquad$ $$\pi^{\ast}(s) = argmax_{a} \left[r(s, a) + \gamma V^{\ast}(\delta(s, a)) \right]$$<br><br>
의미를 다시 보자면, 'state $$s$$에서의 $$optimal \; policy$$란 이 state에서 어떤 action $$a$$를 취했을 때 "_immediate reward $$r(s, a)$$와 그 action으로 도달하게 되는 후속 state $$\delta(s, a)$$의 maximum discounted cumulative reward $$\gamma V^{\ast}(\delta(s, a))$$의 합_"이 최대가 되도록 하는 $$policy$$를 말한다'는 뜻입니다.<br>
그런데 이 식은 $$r(\cdot)$$과 $$\delta(\cdot)$$에 대한 정보가 없이는 풀 수가 없습니다. 그래서 이 term들을 없애기 위해 고안된 것이 다음의 Q-function입니다:<br>
$\qquad$ $$Q(s, a) \equiv r(s, a) + \gamma V^{\ast}(\delta(s, a))$$<br><br>
스포일러를 좀 풀자면, 이 Q-function은 이후 Q-estimate인 $$\hat{Q}$$ term을 이용하여 recursive한 식으로 만듭니다. 그리고 $$\hat{Q}$$가 $$Q$$로 수렴되도록 모든 state의 모든 action의 pair를 stochastically 수행한다는 것이 Q-learning입니다. 이 과정을 식으로 나타내면 다음과 같습니다:<br>
$\qquad$ $$\pi^{\ast}(s) = argmax_a Q(s, a)$$<br>
위 식은 $$optimal \; policy$$의 예전 식을 그냥 $$Q$$ term을 넣어서 rewrite한 것에 불과합니다.<br>
$\qquad$ $$V^{\ast}(s) = max_{a'}Q(s, a')$$<br>
이건 state $$s$$에서의 maximum discounted cumulative reward를 뜻하는 $$V^{\ast}$$를 식으로 쓴 것이고요.<br><br>
이제 위의 Q-function을 다음과 같이 다시 쓸 수 있습니다:<br>
$\qquad$ $$Q(s, a) \equiv r(s, a) + \gamma max_{a'} Q(\delta(s, a), a')$$<br>
이 식을 estimated Q-function인 $$\hat{Q}$$로 바꿔쓰면 (next state는 $$\delta(s, a) = s'$$ 라고 쓰고요), 드디어 최종 식이 나옵니다:<br>
$\qquad$ $$\hat{Q}(s, a) \leftarrow r(s, a) + \gamma max_{a'} \hat{Q}(s', a')$$<br><br>
이제 아래의 pseudocode대로 충분히 여러차례 반복수행을 하게되면 Q-estimation이 real Q에 수렴하게 됩니다:<br>
- - -
<<Q learning algorithm>><br>
For each $$s, a$$ initialize the table entry $$\hat{Q}(s, a)$$ to zero.<br>
Observe the current state $$s$$<br>
Do forever:
- Select an action $$a$$ and execute it<br>
- Receive immediate reward $$r$$<br>
- Observe the new state $$s'$$<br>
- Update the table entry for $$\hat{Q}(s, a)$$ as follows:<br>
      $\qquad$ $$\hat{Q}(s, a) \leftarrow r(s, a) + \gamma max_{a'} \hat{Q}(s', a')$$
- $s \leftarrow s'$

- - -


Toy example을 어떻게 state diagram으로 만들고, 그에 따른 reward table을 만들고, Q table을 학습하는 지는 [여기](http://mnemstudio.org/path-finding-q-learning-tutorial.htm)를 참고하시면 될 것 같습니다. 보다 더 일반적인 개념으로 temporal difference learning도 있고, Q learning에 randomness를 가미하는 $\epsilon$-greedy 등등의 방법들에 대해서는 뒤에 필요한 순간에 설명하도록 하고 지금은 생략합니다. 


**Deep Q Networks**<br><br>
간단하지만 powerful한 Q learning이 여러 toy problem에서 좋은 성능을 보였음에도 많이 사용될 수 없었던건 state와 action의 dimension이 커질수록 (특히 state) Q table의 dimension이 너무 커진다는 문제 때문입니다. 예를 들어 조악한 해상도를 가진 Atari 게임만 해도 Q table의 dimension이 7000x4(상하좌우로 움직이는 경우)나 되는데, camera 입력을 받는 자율주행 자동차 같은건 테이블 크기가 엄청나겠지요. Reward는 상대적으로 매우 sparse 해 질 것이고, 이걸 다 채우도록 학습을 하려면 엄청나게 많은 episode (또는 trajectory이나 rollout이라고 부름)가 필요할 것입니다. 한 마디로 불가능합니다.


이후 DNN이 high-dimensional real-world problem들을 성공적으로 해결하게 되고, Q learning에도 DNN 방법론을 접목하게 된 것이 Deep Q Networks (DQN) 입니다. DQN은 $state$를 입력하면 각 possible $action$에 대한 Q value들이 출력되는 DNN을 학습하여 기존의 Q table을 대체하였습니다.<br>

![Fig2](https://jiryang.github.io/img/dqn.jpg "Deep Q Network"){: width="80%"}{: .aligncenter}


_Experience Replay_<br>
Q learning이 current episode의 연속적인 state-action pair에 대해 매번 Q table을 업데이트하는 방식이다보니 DQN도 episode가 진행되는 동안 계속해서 weight update를 하게 되는데, 이러면 매 episode의 sequence에 대해 DQN이 overfit되는 문제가 발생합니다. 이러한 consecutive sample 사이의 correlation을 없애기 위해 _Experience Replay_ 라는 기법이 사용됩니다. _Experience Replay_ 는 DQN을 on-policy로 업데이트하는 대신, 매번 consecutive하게 발생하는 experience를 별도의 replay memory에 저장해 두었다가 나중에 off-policy 방식으로 random하게 추출하면서 DQN을 업데이트하는 기법입니다. 이를 이용하면 consecutive expericne의 correlation도 줄일 수 있을 뿐더러, 한 experience가 DQN을 한 번 업데이트하고 사라져버리는 것이 아니라 재활용될 수 있고, mini-batch를 사용한 학습 속도 개선에도 기여할 수 있다는 부가적인 장점도 가지고 있습니다.<br><br>


_$\epsilon$-Greedy_<br>
$\epsilon$-greedy는 DQN에만 해당하는 것은 아니고 traditional Q learning에 적용되는 방법으로, Q learning이 greedy한 방식으로 exploitation에만 집중하는 것을 막기 위해 $\epsilon$ 만큼의 확률로 action을 random하게 선택하도록 하는 방법입니다. 알고리즘 수렴을 위해 $\epsilon$은 iteration을 더해가면서 줄여나가게끔 디자인합니다.<br><br>

- - -
<<Deep Q learning algorithm (w/ Experience Replay)>><br>
Initialize replay memory $\mathcal{D}$ to capacity $N$<br>
Initialize action-value function $Q$ with random weights<br>
**for** episode = 1, $M$ **do**<br>
$\qquad$ Initialize sequence $s_1 = \lbrace x_1 \rbrace$ and preprocessed sequenced $\phi_1 = \phi(s_1)$<br>
$\qquad$ **for** $t=1, T$ **do**<br>
$\qquad$ $\qquad$ With probability $\epsilon$ select a random action $a_t$<br>
$\qquad$ $\qquad$ otherwise select $a_t = max_a Q^{\ast}(\phi(s_t), a; \theta)$<br>
$\qquad$ $\qquad$ Execute action $a_t$ in emulator and observe reward $r_t$ and image $x_{t+1}$<br>
$\qquad$ $\qquad$ Set $s_{t+1} = s_t, a_t, x_{t+1}$ and preprocess $\phi_{t+1} = \phi(s_{t+1})$<br>
$\qquad$ $\qquad$ Store transition ($\phi_t, a_t, r_t, \phi_{t+1}$) in $\mathcal{D}$<br>
$\qquad$ $\qquad$ Sample random minibatch of transitions ($\phi_j, a_j, r_j, \phi_{j+1}$)<br>
$\qquad$ $\qquad$ Set $y_j = r_j$ for terminal $\phi_{j+1}$<br>
$\qquad$ $\qquad$ Set $y_j = r_j + \gamma max_{a'} Q(\phi_{j+1}, a'; \theta)$ for non-terminal $\phi_{j+1}$<br>
$\qquad$ $\qquad$ Perform a gradient descent step on ${(y_j - Q(\phi_j, a_j; \theta))}^2$<br>
$\qquad$ **endfor**<br>
**endfor**
- - -
맨 뒷부분의 gradient descent step은 결국 target (current) Q와 estimated Q 사이의 mean-squared error를 구한 것이라고 보셔도 무방합니다.
~~~
...
loss = self.MSE_loss(curr_Q, expected_Q.detach())
...
~~~


성능 향상을 위해 사용된 혹은 사용 가능한 추가적인 기법들 및 DQN variant들을 몇 개 소개하고 DQN을 마무리할까 합니다.<br>


_Soft Update (Target Network)_<br>
이건 Q table이 네트워크에 녹아있는 DQN pseudocode가 아니라 Q update가 explicit하게 표현된 Q learning을 가지고 설명하는게 더 이해가 쉽겠습니다. 저 위에 Q learning pseudocode를 보시면 estimated Q를 추정하여 업데이트시켜주는 부분이 $\hat{Q}(s, a) \leftarrow r(s, a) + \gamma max_{a'} \hat{Q}(s', a')$ 으로 표현되어 있습니다. 우변의 estimation을 가지고 좌변의 target을 update하는 방식인데 $\hat{Q}$ term이 양변에 동일하게 들어가 있기 때문에, 매번 target이 update 될때마다 estimate의 값이 oscillate하게 되어서 $\hat{Q}$가 수렴하기 어려운 문제가 생깁니다. 그래서 별도의 target Q table (DQN의 경우에는 별도의 target Q network이 되겠죠)을 두고 간헐적으로 (원래 Q network보다 드문드문) 업데이트를 함으로써 Q network 수렴을 돕는다는 개념입니다. Target Q network의 update frequency가 아니라 update magnitude를 조절하여 유사한 효과를 노린 경우도 있었습니다 ([링크](https://arxiv.org/pdf/1509.02971.pdf)). 여기선 $\hat{Q} = \tau Q + (1-\tau)\hat{Q}, \tau = 0.001$ 와 같이 target Q의 update량을 미세하게 만들었습니다.<br><br>


이 외에도 DQN으로 Atari game들을 수행하면서 여러 task를 동일한 parameter (learning rate)로 학습하고 성능을 높이기 위해 각 task의 reward를 scale해주는 _Clipping Rewards_ 나 computational cost을 낮추어 더 많은 experience를 확보하기 위한 _Frame Skipping_ 와 같은 트릭들이 사용되었습니다.<br>


**Double DQN (DDQN)**<br>
Q learning은 current state에 대해 Q table의 current estimation ($\hat{Q})에서 reward를 maximize하는 action을 선택하고, 그 max reward를 decay시켜 next Q를 update했습니다:<br>
$\qquad$ $$\hat{Q}(s, a) \leftarrow r(s, a) + \gamma max_{a'} \hat{Q}(s', a')$$<br>

DQN도 마찬가지였죠:<br>
$\qquad$ Select $$a_t = max_a Q^{\ast}(\phi(s_t), a; \theta)$$<br>
$\qquad$ $$...$$<br>
$\qquad$ Perform a gradient descent step on $${(y_j - Q(\phi_j, a_j; \theta))}^2$$<br>

'모든 state가 관찰 가능하고, 모든 state에 대한 모든 action이 수행 가능한' perfect environment에서라면 그다지 문제가 되지 않을 수도 있으나, 실제 학습 상황에서는 sensory값이 누락된다거나 environment에 변화가 생긴다거나 $\epsilon$-greedy처럼 non-deterministic하게 action을 수행하게 되는 noisy한 environment일 가능성이 높습니다. Perfect environment라면 stochastic하게 optimal Q value로 수렴이 될 가능성이 높지만, 그렇지 않은 경우 Q table의 값이 몇몇 경우의 잘못 계산된 $max \hat{Q}$ 값에 의해 suboptimal하게 수렴되거나 oscillate하는 경우가 발생할 수 있습니다. 이를 방지하기 위해 기존 Q network의 delayed copy를 별도로 두고, action selection은 원래의 Q network에서 하되, 그 action으로 인한 reward는 이 delayed copy에서 가져오는 방식이 고안되었고, 이 delayed copy때문에 Double Deep Q Network (DDQN) 이라는 이름이 붙게 되었습니다.<br>

이 두 번째 network를 delayed copy라고 부르는 이유는, primary Q network이 매 state마다 update되는 반면에 두 번째 network은 episode가 끝난 뒤 primary network와 자기 자신의 차이를 줄이는 방향으로 학습하기 때문입니다.



[Fig3](https://jiryang.github.io/img/dqn_atari_result.png "DQN vs Human on Atari Games"){: width="100%"}


[Fig4](https://jiryang.github.io/img/dqn_atari_master.gif "DQN on Atari Breakout"){: width="50%"}


