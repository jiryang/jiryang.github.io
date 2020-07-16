---
layout: post
title:  "Deep Neuroevolution - part4"
date:   2020-07-15 01:00:01
categories: ReinforcementLearning
tags: reinforcement_learning policy_gradient
excerpt: Policy-Based Reinforcement Learning
mathjax: true
---

[지난 포스트](https://jiryang.github.io/2020/07/14/neuroevolution03/)에서 살펴본 Q learning의 $optimal \; policy$란 cumulative reward를 maximize하는 것이므로 모든 state의 모든 action에 대한 quality value를 계산해두면 어느 state에서건 goal에 이르는 optimal (cumulative reward를 maximize하는) action을 구할 수 있다'로 요약할 수 있습니다. 


하지만 real-world task에 적용하기에는 여러가지 문제가 있습니다. 우선 앞서 traditional Q learning의 단점으로 지적되었던 state-action space dimension 문제입니다. DQN으로 state space가 Atari game 정도로 확장된 task들에도 적용이 가능해지긴 했지만, 여전히 higher dimensional continuous state space에는 적용이 어렵고, 특히 Atari game과 달리 action space가 continuous한 경우는, discretization과 같은 트릭을 쓴다해도 scaling에 큰 제약이 있습니다. State 및 action space dimension이 늘어나면 Q network 자체 뿐만 아니라 성능 개선을 위해 추가했던 replay memory의 사이즈도 폭발적으로 증가하게 될 것입니다. 또한, exploration을 강화해서 학습을 '넓게'하기 위한 목적으로 추가한 $\epsilon$-greedy도 문제가 될 수 있는데요, optimal policy가 deterministic한 경우 작긴 하지만 계속해서 $\epsilon$만큼의 확률로 random action selection을 하게 되면 수렴 및 performance에 악영향이 있을 수 있습니다. 이러한 이유로 낭비스럽게 모든 state-action pair에 대한 Q value를 학습한 다음 거기서 optimal policy를 간접적으로 구해서 쓰는 방법 대신, input state에 대한 최적의 policy를 바로 학습하는 policy gradient method가 고안되었습니다.

![Fig1](https://jiryang.github.io/img/dqn_6s191.PNG "DQN at a glance"){: width="100%"}{: .aligncenter}


![Fig2](https://jiryang.github.io/img/pg_6s191.PNG "PG at a glance"){: width="100%"}{: .aligncenter}

_* 이미지는 MIT 6.S191 lecture slide에서 가져왔습니다. 대신 [유튜브 링크](https://www.youtube.com/watch?v=nZfaHIxDD5w&t=1937s) 공유합니다. 잘 준비된 intro 강의라서 이해가 쉬우니 꼭 보시길 권합니다._


Policy gradient의 장점은 그 결과가 current state에 대한 각 action의 확률로 나오기 때문에 optimal policy가 deterministic한 경우라면 낮은 확률으로라도 randomness를 강제했던 value-based method의 $\epsilon$-greedy와 달리 stochastically deterministic하게 수렴하게 되며, optimal policy가 arbitrary한 경우에도 probability-based로 동작하기 때문에 대응이 가능하다는 점을 들 수 있습니다. 두 번째 경우를 좀 더 설명하자면, 예를들어 포커 게임을 학습한 경우 낮은 패를 쥐었을 때 Q learning과 같은 value-based 방법은 $argmax$로 도출한 optimal policy가 100% fold로 나오게 되는 반면 policy gradient는 가끔씩 블러핑을 할 수도 있습니다. 또한 앞서 설명한대로 모든 state-action space를 탐색하지 않고 probabilistically greedy하게 필요한 action을 선택하는 policy space만을 탐색하기 때문에 학습이 효과적입니다 (faster with fewer parameters). 또한 policy의 distribution이 Gaussian이라 가정하면 action space를 mean과 variance로 modeling할 수도 있게 됩니다. 즉, policy gradient를 DNN으로 구현했다면 이 output이 action들의 probability vector가 될 필요가 없고 action space를 나타내는 mean(zero-mean이라 가정하고 mean의 shift 값을 출력하면 되겠죠)과 variance만 출력해도 된다는 뜻입니다. 이렇게 되면 action space가 continuous하게 방대한 경우에도 modeling이 가능해집니다.

![Fig3](https://jiryang.github.io/img/model_continuous_action_space.PNG "PG Modeling Continuous Action Space"){: width="100%"}{: .aligncenter}


Value based 대비 policy gradient 방식의 단점은 environment의 작은 변화에도 성능이 영향을 받는다는 것을 들 수 있습니다. Value table을 학습한다는건 당장 optimal policy를 구하는 데는 쓰이지 않더라도 모든 state-action space의 lookahead table을 만들어둔다는 것이라고 생각할 수도 있는데요, 그렇기 때문에 environmental change에 어느정도 resilience를 가집니다. 하지만 current environment에서 optimal한 policy를 찾는데 최적화된 policy network는 작은 environmental change에도 학습을 새로 해야 합니다.


여타 RL과 마찬가지로 policy gradient에서도 _expected_ reward를 maximize하는 것이 그 목표입니다.<br>
$\qquad$ $$J(\theta) = \mathbb{E}_{\pi}\left[ r(\tau) \right]$$

$\qquad$ $\qquad$ $\pi$ or $\pi_{\theta}$: policy (parameterized by $\theta$)

$\qquad$ $\qquad$ $r(\tau)$: total reward for a given episode $\tau$
<br>


우리의 목표는 위의 $J$를 최대화하는 parameter $\theta$를 찾는 것이겠지요. 네트워크로 구현된 경우라면 저 $\theta$는 weights가 될 것입니다. Target $Q$와 estimated $Q$ 사이의 mean-quared error를 loss로 놓고 gradient descent search를 했던 Q learning과 달리, policy-based RL의 objective function은 expected reward이기 때문에 minimize하는 것이 아니라 maximize를 해야합니다. Update rule이 다음과 같이 되겠죠:<br><br>
$\qquad$ $$\theta_{t+1} = \theta_t + \alpha \nabla J(\theta_t)$$

$\pi$가 policy의 probability로 표현되는 policy-based에서는 (특히 continous라면 더욱) _expected_ reward를 다음과 같이 integral로 표현할 수 있습니다:<br><br>
$\qquad$ $$J(\theta) = \mathbb{E}_{\pi}\left[ r(\tau) \right] = \int \pi(\tau)r(\tau)$$

$\qquad$ $$\nabla J(\theta) = \nabla  \mathbb{E}_{\pi}\left[ r(\tau) \right] = \nabla \int \pi(\tau)r(\tau)d\tau$$

$\qquad$ $\qquad$ $$= \int \nabla \pi(\tau)r(\tau)d\tau$$

$\qquad$ $\qquad$ $$= \int \pi(\tau) \nabla ln \; \pi(\tau) r(\tau)d\tau \quad (\because ln \; \pi(\tau) = \frac{1}{\pi(\tau)})$$

$\qquad$ $\qquad$ $$= \mathbb{E}_{\pi}\left[ r(\tau) \nabla ln \; \pi(\tau) \right]$$

<br>

마지막 식을 글로 풀어쓰면 '_expected_ reward의 미분값은 reward $\times$ (policy($\pi_{\theta}$)에 로그를 취한 값의 gradient)와 같다' 인데요, 이것이 바로 **_Policy Gradient Theorem_** 입니다.

_* Policy Gradient theorem의 증명은 여러 방식으로 가능한데요, 또다른 증명 한 가지를 [링크](https://lilianweng.github.io/lil-log/2018/04/08/policy-gradient-algorithms.html)로 대신합니다._<br>

이렇게 policy gradient에 $ln$을 취한 값은 product($\prod$)로 표현되던 episode 내 policy를 sum으로 바꿔주고, $\theta$와 무관한 initial state의 probability 및 state transition probability term을 제거시켜 주어 derivative of _expected_ reward를 episode 내의 각 step의 policy probability만으로 간소화 시켜주는 효과를 낳아, 드디어 policy의 미분값만으로 backpropagation을 이용해서 (그리고 reward 값과 곱해야죠) gradient의 contribution을 구할 수 있게 됩니다:<br><br>
$\qquad$ $$\pi_{\theta}(\tau) = \mathcal{P}(s_0) \prod^T_{t=1} \pi_{\theta}(a_t \mid s_t)p(s_{t+1}, r_{t+1} \mid s_t, a_t)$$

$\qquad$ $$ln \;\pi_{\theta}(\tau) = ln \; \mathcal{P}(s_0) + \sum^T_{t=1} ln \; \pi_{\theta}(a_t \mid s_t) + \sum^T_{t=1} ln \; p(s_{t+1}, r_{t+1} \mid s_t, a_t)$$
<br>

이 과정 이후 남은 식은 다음과 같습니다:<br>
$\qquad$ $$\nabla \mathbb{E}_{\pi_{\theta}} \left[ r(\tau) \right] = \mathbb{E}_{\pi_{\theta}} \lbrack r(\tau) \left( \sum^T_{t=1} \nabla ln \; \pi_{\theta} (a_t \mid s_t) \right) \rbrack$$
<br>

이제 $r(\tau)$를 $\sum$ 안에 넣으면 해당 step부터의 cumulative reward인 $G_t$로 바꾸어 표기할 수 있으며, REINFORCE 알고리즘의 최종 update rule이 완성됩니다:<br>
$\qquad$ $$\nabla \mathbb{E}_{\pi_{\theta}} \left[ r(\tau) \right] = \mathbb{E}_{\pi_{\theta}} \lbrack \left( \sum^T_{t=1} G_t \nabla ln \; \pi_{\theta} (a_t \mid s_t) \right) \rbrack \quad (\because G_t = \sum^T_{t=1} R_t)$$
<br><br>


**REINFORCE (Monte-Carlo Policy Gradient)**<br>

Pseudocode부터 보겠습니다:<br>
- - -
<<REINFORCE algorithm>><br>
Initialize the policy parameter $\theta$ at random.<br>
Do forever:
- Generate $N$ episodes ($\tau_1 \sim \tau_N$) following the policy. Each episode $\tau_i$ has the following sequence: $s_{i, 0}, a_{i, 0}, r_{i, 1}, s_{i, 1}, ..., s_{i, T-1}, a_{i, T-1}, r_{i, T}, s_{i, T}$<br>
- Evaluate the gradient using these samples:<br>
$\qquad$ $$\nabla_{\theta}J(\theta) \approx \frac{1}{N} \sum^N_{i=1} \left( \sum^{T_i - 1}_{t=0} G_{i, t} \nabla ln \; \pi_{\theta} (a_{i, t} \mid s_{i, t}) \right)$$<br>
$\qquad$ where $G_{i, t}$ for trajectory $\tau_i$ at time $t$ is defined as the cumulative rewards from the beginning of the episode:<br>
$\qquad$ $$G_{i, t} = \sum^{T_i - 1}_{t'=0} r(s_{i, t'}, a_{i, t'}))$$<br>
  - $\theta \leftarrow \theta + \alpha \nabla_{\theta}J(\theta)$

- - -

이는 Monte-Carlo 방식의 REINFORCE로, on-policy로 동작하기 때문에 episode들을 수행한 뒤 update하는 방식이 아니라 매 step마다 online update를 하기 때문에 $i$ term이 추가되었습니다 (여전히 future reward는 online으로 알 방법이 없으니 일단 episode를 끝까지 돌려서 각 step의 reward를 구한 다음, 다시 해당 episode의 매 step을 복기하면서 step-wise gradient update를 하면 됩니다). 충분히 큰 $N$ 횟수만큼 돌리면 정답에 수렴하게 될 것이기 때문에 등호 대신에 $\approx$ 기호를 사용했습니다.


위와 같은 update rule을 가지는 vanilla REINFORCE는 policy가 probabilistic하게 결정되기 때문에 특정 episode의 특정 state에서 서로 다른 action을 선택할 가능성이 늘 있습니다. 그런데 optimal vs suboptimal policy에 대한 $G_t$값의 차이가 지나치게 들쭉날쭉하다면 (variance가 크다면), $G_t$가 update rule의 매 gradient에 곱해지는 값이기 때문에 학습이 수렴되는 것을 어렵게 만듭니다. 예를 들면 아래의 그래프처럼 수렴에 수많은 iteration이 필요하고 variance가 큰 것을 볼 수 있습니다 (David Silver의 RL Ch.7 강의자료이며, iteration 단위가 million임).이를 좀 완화하기 위해 cumulative reward ($G_t$)를 구할 때 discount rate ($\gamma$)를 추가하였지만 여전히 variance를 충분히 줄여주지는 못합니다. 이러한 variance 문제를 줄여주기 위한 몇 가지 방법이 고안되었습니다:<br>

![Fig4](https://jiryang.github.io/img/vanilla_pg_convergence.PNG "Example of Convergence Graph of Monte-Carlo REINFORCE"){: width="50%"}{: .aligncenter}


**_Causality (Reward-to-go)_**<br>
앞서 $G_t$를 $t=0 \sim T$ ($T$: time of episode termination) 까지의 reward의 합으로 계산하였는데요, 이 cumulative reward는 'current state $s$에서 action $a$를 취할 때 받을 immediate reward 및 future reward의 총합'이기 때문에 과거에 이미 받은 reward를 현재의 cumulative reward에 더할 필요가 없습니다. 즉 $$G_{i, t} = \sum^{T_i - 1}_{t'=0} r(s_{i, t'}, a_{i, t'})$$가 $$G_{i, t} = \sum^{T_i - 1}_{t'=t} r(s_{i, t'}, a_{i, t'})$$ 로 바꿀 수 있으며, 이로 인해 전체적으로 gradient에 곱해지는 값의 magnitude를 떨어뜨려 variance 문제를 완화시킬 수 있습니다.<br>


**_Discount rate_**<br>
앞서 Q learning에서도 보셨듯이 future reward에 discount rate를 곱해서 $G_t$를 discounted cumulative reward로 만들 수 있으며, 이것 또한 variance를 줄여줍니다. 


_Causality_ 와 _Discount rate_ trick을 적용한 gradient estimate은 다음과 같이 변경됩니다:<br>
$\qquad$ $$\nabla_{\theta}J(\theta) \approx \frac{1}{N} \sum^N_{i=1} \sum^{T_i - 1}_{t=0} \nabla_{\theta} ln \; \pi_{\theta} (a_{i, t} \mid s_{i, t}) \times \left( \sum^{T_i - 1}_{t'=t} r(s_{i, t'}, a_{i, t'}) \right)$$<br>


**_Baseline_**<br>
위의 **policy gradient theorem**은 reward에서 action($\theta$)에 dependent하지 않은 어떠한 식을 차감하더라도 상수화해서 소거가 가능하기 때문에 variance를 줄여주기 위한 arbitrary function을 넣어줄 수 있습니다. 모든 policy가 $\ge$ 0인 대부분의 경우 "좋은" policy든 "나쁜" policy든 reward를 증가시키게 되는데, 이에 비해 "나쁜" policy가 reward를 감소시키도록 하면 variance도 줄어들고 수렴도 쉬울 것입니다. 이같은 역할을 위한 function을 _baseline_ 이라 하고, _baseline_ 이 적용된 update rule은 다음과 같이 바뀝니다 (_Causality_ 와 _Discount rate_ 도 적용):<br>
$\qquad$ $$\nabla_{\theta}J(\theta) \approx \frac{1}{N} \sum^N_{i=1} \sum^{T_i - 1}_{t=0} \nabla_{\theta} ln \; \pi_{\theta} (a_{i, t} \mid s_{i, t}) \times \left( \sum^{T_i - 1}_{t'=t} r(s_{i, t'}, a_{i, t'}) - b(s_t) \right)$$<br><br>


아무 function이나 _baseline_ 으로 쓸 수는 있지만 몇 가지 대표적인 예는 다음과 같습니다:
- Constant baseline: 모든 episode $\tau$의 final reward의 평균을 baseline으로 차감<br>
$\qquad$ $$b = \mathbb{E} \lbrack R(\tau) \rbrack \approx \frac{1}{N} \sum^N_{i=1} R(\tau^{(i)})$$
- Optimal Constant baseline: 수학적으로 variance ($$Var \lbrack x \rbrack = E \lbrack x^2 \rbrack - E {\lbrack x \rbrack}^2 $$)를 최소화하는 값을 계산한 optimal 값이지만 성능 개선 정도에 비해 computational burden이 심해서 자주 사용되지는 않음<br>
$\qquad$ $$b = \frac{\sum_i (\nabla_{\theta} log \; P(\tau^{(i)}; \theta)^2)R(\tau^{(i)})}{\sum_i (\nabla_{\theta} log \; P(\tau^{(i)}); \theta)^2}$$<br>
- Time-dependent baseline: episode 기준으로 reward를 계산하여 averaging을 하는 것이 아니라, 각 episode 내의 모든 step(state-action pair)들에 대해 reward를 구해 평균을 낸 것으로, _Causality (Reward-to-go)_ 적용 가능 (수식은 _Causality_ 적용)<br>
$\qquad$ $$b_t = \frac{1}{N} \sum^N_{i=1} \sum^{T-1}_{t'=t} r(s_{i, t'}, a_{i, t'})$$
- State-dependent expected return: episode나 time이 아니라 특정 state에 dependent한 reward (현재 policy에 의하면 state $t$에서는 평균 얼마만큼의 reward를 주는가)를 계산<br>
$\qquad$ $$b(s_t) = \mathbb{E} \lbrack r_t + r_{t+1} + r_{t+2} + ... + r_{T-1} \rbrack = V^{\pi}(s_t)$$
<br><br>


Policy-gradient도 variant들을 몇 개 소개합니다.<br>

**Actor-Critic Method**

위의 baseline 중 마지막 state-dependent expected return은 **Actor-Critic Method**라고 합니다. $V(s)$를 학습하여

**Trust Region Policy Optimization (TRPO)**

**Proximal Policy Optimization (PPO)**