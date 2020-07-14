---
layout: post
title:  "Deep Neuroevolution - part4"
date:   2020-07-14 01:00:01
categories: ReinforcementLearning
tags: reinforcement_learning policy_gradient
excerpt: Policy-Based Reinforcement Learning
mathjax: true
---

앞서 살펴본 Q learning의 $optimal \; policy$란 cumulative reward를 maximize하는 것이므로 모든 state의 모든 action에 대한 quality value를 계산해두면 어느 state에서건 goal에 이르는 optimal (cumulative reward를 maximize하는) action을 구할 수 있다'로 요약할 수 있습니다. 하지만 real-world task에 적용하기에는 여러가지 문제가 있습니다. 우선 앞서 traditional Q learning의 단점으로 지적되었던 state-action space dimension 문제입니다. DQN으로 state space가 Atari game 정도로 확장된 task들에도 적용이 가능해지긴 했지만, 여전히 higher dimensional continuous state space에는 적용이 어렵고, 특히 Atari game과 달리 action space가 continuous한 경우는, discretization과 같은 트릭을 쓴다해도 scaling에 큰 제약이 있습니다. Q network 자체 뿐만 아니라 성능 개선을 위해 추가했던 replay memory의 사이즈도 폭발적으로 증가하게 될 것이기 때문입니다. 또한, exploration을 강화해서 학습을 '넓게'하기 위한 목적으로 추가한 $\epsilon$-greedy도 문제가 될 수 있는데요, optimal policy가 deterministic한 경우 작긴 하지만 계속해서 $\epsilon$만큼의 확률로 random action selection을 하게 되면 수렴 및 performance에 악영향이 있을 수 있습니다. 이러한 이유로 낭비스럽게 모든 state-action pair에 대한 Q value를 학습한 다음 거기서 optimal policy를 구해서 쓰는 간접적인 방법 대신, input state-action pair에 대한 policy를 바로 학습하는 policy gradient method가 고안되었습니다.


Policy gradient의 장점은 optimal policy가 deterministic한 경우라면 ($\epsilon$-greedy와 달리) stochastically deterministic하게 수렴하게 되며, optimal policy가 arbitrary한 경우에도 probability-based로 동작하기 때문에 대응이 가능하다는 점을 들 수 있습니다. 두 번째 경우를 좀 더 설명하자면, 예를들어 포커 게임을 학습한 경우 Q learning과 같은 value-based 방법은 낮은 패를 쥐어서 optimal policy가 fold로 나오는 경우에도 policy gradient 방법은 낮은 확률로 블러핑을 할 수도 있습니다. 또한 앞서 설명한대로 모든 state-action space를 탐색하지 않고, probabilistically greedy하게 필요한 action을 선택하는 policy space만을 탐색하기 때문에 학습이 효과적입니다 (faster with fewer parameters).


Value based 대비 policy gradient 방식의 단점은 environment의 작은 변화에도 성능이 영향을 받는다는 것을 들 수 있습니다. Value table을 학습한다는건 당장 optimal policy를 구하는 데는 쓰이지 않더라도 모든 state-action space의 lookahead table을 만들어둔다는 것이라 생각할 수 있는데요, 그렇기 때문에 environmental change에 어느정도 resilience를 가집니다. Current environment에서 optimal한 policy를 찾는데 최적화된 policy network는 작은 environmental change에도 학습을 새로 해야 합니다.


여타 RL과 마찬가지로 policy gradient에서도 _expected_ reward를 maximize하는 것이 그 목표입니다.<br>
$\qquad$ $$J(\theta) = \mathbb{E}_{\pi}\left[ r(\tau) \right]$$

$\qquad$ $\pi$ or $\pi_{\theta}$: policy (parameterized by $\theta$)

$\qquad$ $r(\tau)$: total reward for a given trajectory $\tau$


우리의 목표는 위의 $J$를 최대화하는 parameter $\theta$를 찾는 것이겠지요. 네트워크로 구현된 경우라면 저 $\theta$는 weights가 될 것입니다. Target $Q$와 estimated $Q$ 사이의 $MSE$를 loss로 놓고 gradient descent search를 했던 Q learning과 달리, policy-based RL의 objective function은 expected reward이기 때문에 minimize하는 것이 아니라 maximize를 해야합니다. Update rule이 다음과 같이 되겠죠:<br>
$\qquad$ $$\theta_{t+1} = \theta_t + \alpha \nabla J(\theta_t)$$

$\pi$가 policy의 probability로 표현되는 policy-based에서는 (특히 continous라면 더욱) _expected_ reward를 다음과 같이 integral로 표현할 수 있습니다:<br>
$\qquad$ $$J(\theta) = \mathbb{E}_{\pi}\left[ r(\tau) \right] = \int \pi(\tau)r(\tau)$$

$\qquad$ $$\nabla J(\theta) = \nabla  \mathbb{E}_{\pi}\left[ r(\tau) \right] = \nabla \int \pi(\tau)r(\tau)d\tau$$

$\qquad$ $$= \int \nabla \pi(\tau)r(\tau)d\tau$$

$\qquad$ $$= \int \pi(\tau) \nabla ln \; \pi(\tau) r(\tau)d\tau \quad (\because ln \; \pi(\tau) = \frac{1}{\pi(\tau)})$$

$\qquad$ $$= \mathbb{E}_{\pi}\left[ r(\tau) \nabla ln \; \pi(\tau) \right]$$

마지막 식을 글로 풀어쓰면 '_expected_ reward의 미분값은 reward와 policy($\pi_{\theta}$)에 로그를 취한 값의 gradient와의 곱과 같다' 인데요, 이것이 바로 **_Policy Gradient Theorem_** 입니다.

_* Policy Gradient theorem의 증명은 굉장히 여러 방식으로 가능한데요, 또다른 증명 한 가지를 [링크](https://lilianweng.github.io/lil-log/2018/04/08/policy-gradient-algorithms.html)로 대신합니다._<br>

이렇게 policy gradient에 $ln$을 취한 값은 product($\prod$)로 표현되던 episode 내 policy를 sum으로 바꿔주고, $\theta$와 무관한 initial state의 probability 및 state transition probability term을 제거시켜주어 derivative of _expected_ reward를 episode 내의 각 step의 policy probability만으로 간소화 시켜주는 효과를 낳아, 드디어 policy의 미분값만으로 backpropagation을 이용해서 (그리고 reward 값과 곱해야죠) gradient의 contribution을 구할 수 있게 됩니다:<br>
$\qquad$ $$\pi_{\theta}(\tau) = \mathcal{P}(s_0) \prod^T_{t=1} \pi_{\theta}(a_t \mid s_t)p(s_{t+1}, r_{t+1} \mid s_t, a_t)$$

$\qquad$ $$ln \;\pi_{\theta}(\tau) = ln \; \mathcal{P}(s_0) + \sum^T_{t=1} ln \; \pi_{\theta}(a_t \mid s_t) + \sum^T_{t=1} ln \; p(s_{t+1}, r_{t+1} \mid s_t, a_t)$$

이 과정 이후 남은 식은 다음과 같습니다:<br>
$\qquad$ $$\nabla \mathbb{E}_{\pi_{\theta}} \left[ r(\tau) \right] = \mathbb{E}_{\pi_{\theta}} \lbrack r(\tau) \left( \sum^T_{t=1} \nabla ln \; \pi_{\theta} (a_t \mid s_t) \right) \rbrack$$

이제 $r(\tau)$를 $\sum$ 안에 넣으면 해당 step부터의 discounted cumulative reward인 $G_t$로 바꾸어 표기할 수 있으며, REINFORCE 알고리즘의 최종 update rule이 완성됩니다:<br>
$\qquad$ $$\nabla \mathbb{E}_{\pi_{\theta}} \left[ r(\tau) \right] = \mathbb{E}_{\pi_{\theta}} \lbrack \left( \sum^T_{t=1} G_t \nabla ln \; \pi_{\theta} (a_t \mid s_t) \right) \rbrack \quad (\because G_t = \sum^T_{t=1} \gamma R_t)$$


위와 같은 update rule을 가지는 vanilla REINFORCE는 policy가 probabilistic하게 결정되기 때문에 특정 episode의 특정 state에서 서로 다른 action을 선택할 가능성이 늘 있습니다. 그런데 optimal vs suboptimal policy에 대한 $G_t$값의 차이가 지나치게 들쭉날쭉하다면 (variance가 크다면), $G_t$가 update rule의 매 gradient에 곱해지는 값이기 때문에 학습이 수렴되는 것을 어렵게 만듭니다. 그래서 **_baseline_**이 도입됩니다. 이 baseline은 action($\theta$)에 dependent하지만 않으면 gradient update rule에서 상수로 취급되어 소거가 가능하기 때문에 이 조건을 만족하면서 variance를 줄여줄 수 있습니다. 대표적인 예로 average performance를 baseline으로 잡아 reward에서 차감시키게 되면 variance를 줄일 수 있게되어 "좋은" policy는 선택될 확률를 (여전히) 증가시키면서도 "나쁜" policy는 선택될 확률을 감소시키는 효과를 발생시켜 모든 policy가 $>0$ 값을 가지는 경우에 비해 수렴이 쉽습니다. Baseline을 적용한 update rule은 다음과 같습니다:<br>
$\qquad$ $$\nabla \mathbb{E}_{\pi_{\theta}} \left[ r(\tau) \right] = \mathbb{E}_{\pi_{\theta}} \lbrack \left( \sum^T_{t=1} (G_t - b) \nabla ln \; \pi_{\theta} (a_t \mid s_t) \right) \rbrack$$


Baseline을 정하는 몇 가지 대표적인 예는 다음과 같습니다:
- Constant baseline: 모든 episode $\tau$의 final reward의 평균을 baseline으로 차감<br>
$\qquad$ $$b = \mathbb{E} \lbrack R(\tau) \rbrack \approx \frac{1}{m} \sum^m_{i=1} R(\tau^{(i)})$$
- Optimal Constant baseline: 수학적으로 variance ($$Var \lbrack x \rbrack = E \lbrack x^2 \rbrack - E {\lbrack x \rbrack}^2 $$)를 최소화하는 값을 계산한 optimal 값이지만 성능 개선 정도에 비해 computational burden이 심해서 자주 사용되지는 않음<br>
$\qquad$ $$b = \frac{\sum_i (\nabla_{\theta} log \; P(\tau^{(i)}; \theta)^2)R(\tau^{(i)})}{\sum_i (\nabla_{\theta} log \; P(\tau^{(i)}); \theta)^2}$$<br>
- Time-dependent baseline: episode 기준으로 reward를 계산하여 averaging을 하는 것이 아니라, 각 episode 내의 모든 state-action pair에 대해 reward를 구해 평균을 낸 것으로, 특정 시간 이후의 state-pair action만을 고려할 수도 있음 (수식에서는 episode마다 $t$ 시점 이후부터의 reward를 계산)<br>
$\qquad$ $$b_t = \frac{1}{m} \sum^m_{i=1} \sum^{T-1}_{k=t} R(s^{(i)}_k, u^{(i)}_k)$$
- State-dependent expected return: episode나 time이 아니라 특정 state에 dependent한 reward (현재 policy에 의하면 state $t$에서는 평균 얼마만큼의 reward를 주는가)를 계산<br>
$\qquad$ $$b(s_t) = \mathbb{E} \lbrack r_t + r_{t+1} + r_{t+2} + ... + r_{T-1} \rbrack = V^{\pi}(s_t)$$