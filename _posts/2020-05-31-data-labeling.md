---
layout: post
title:  "Active Learning: Efficient Data Labeling"
date:   2020-05-31 02:00:00
categories: GeneralML
tags: bigdata labeling active_learning
excerpt: 머신러닝 모델을 학습시키기 위해 어떤 데이터를 얼마나, 어떻게 레이블링 해야 하는가
mathjax: true
---

한 업체에 다녀왔습니다. 이곳도 'AI는 해야겠는데 어떻게 할 지는 모르는' 곳 중 하나였습니다.\
많은 기업들이 비용을 들이자니 palpable한 효과가 잘 그려지지 않고, 손을 놓고 있자니 뒤처질 것 같아 불안한 상태인 것 같습니다.\
그래서 동종 업계의 success나 failure case를 요구하기도 하고, PoC를 해보자고 하는 것이겠지요.


"AI를 언제든 적용할 수 있도록 데이터는 미리 모아두었습니다."\
PoC를 하자는 업체에서 많은 데이터를 받았습니다. 그런데 원하는 모델에 넣기에 레이블이 제대로 되어있지 않습니다. PoC 기간과 예산은 충분하지 않습니다. 어떻게 하면 될까요?


많은 양의 quality data가 필요하다는 점은 현대 supervised deep learning의 커다란 단점으로 지적되고 있습니다.
(AAAI keynote speech에서 얀 르쿤 박사께서도 지적하신 부분이지요 [Self-supervised learning: The plan to make deep learning data-efficient](https://bdtechtalks.com/2020/03/23/yann-lecun-self-supervised-learning/amp/)). 성능을 극대화하기 위해서는 최대한 많은 양의 training data가 필요한데, 모델 학습 시간과 data labeling에 필요한 cost를 생각하면 효율화가 필요합니다. 특히 현업에서 실제로 일어날 가능성이 높은 이러한 trade-off를 효과적으로 해결하기 위한 active learning에 대해 알아보도록 하겠습니다.


앞서 PoC용으로 기업에서 의뢰받은 모델을 만들기 위해 필요한 최소한의 raw data를 labeling 업체에 맡기거나 part-timer들을 써서 label 작업을 마쳤습니다만, 일부 label에 오류가 있기도 하고 추가적인 labeling이 필요한 상태입니다. 주어진 문제의 성격에 따라서 높은 정확도를 가진 labeling을 필요로 하기도 합니다. 예를 들면 이미지 분류기보다는 fraud detector가 더 높은 정확도를 요구할 것이고, 자율주행에 쓰이는 road sign detector는 그보다도 높은 정확도를 요구할 것입니다. Quality labeling을 위해서는 domain expert나 경험이 많은 인력이 필요한데, 이러면 시간과 비용이 더욱이 늘어게 됩니다. 이럴 때 active learning 방법을 사용해서 특정 이상의 model performance를 guarantee하면서 expert annotation을 최소화할 수 있습니다.


Active learning은 "모든 데이터가 모델 학습에 동일한 영향을 미치지 않는다"는 아이디어에서 비롯된 오래된 방법론입니다. 아래 그림은 2-class Gaussian 분포를 가진 400개짜리 dataset에서 random sampling과 active learning 방식을 사용하여 30개씩을 추출하고 binary classifier를 학습한 결과입니다. 동일한 숫자의 데이터로 같은 모델을 학습하였는데도 decision boundary가 다르고, 그 결과 accuracy도 다르게 나온 것을 볼 수 있습니다. Random sampling으로 선택된 데이터는 모 데이터셋의 분포에서 골고루 선택된 것처럼 보이는데, active learning 방식으로 선택된 데이터는 decision boundary 부근에 몰려있는 차이가 있네요. 모델의 decision boundary 학습에 영향력이 큰 데이터들을 선택한다면 같은 숫자의 데이터라도 효과적인 학습을 할 수 있는 것 같지요? 다르게 표현하면, 무조건 data label 작업을 할 것이 아니라 학습에 더 도움이 될 것 같은 데이터만 전문가(active learning에서는 이를 _human oracle_ 이라고도 합니다)에게 의뢰하여 labeling을 하면 된다는 의미이기도 합니다. 이것이 사실이라면 학습에 별로 영향이 없는 데이터는 굳이 labeling을 하지 않아도 될테니 시간과 비용을 줄일 수 있을 것입니다. 이렇게 '중요한' 데이터들에 대해서만 label을 추가해서 모델을 학습하면 목표한 성능을 달성하는데 필요한 학습 시간도 빨라질 것입니다. 

![Fig1](https://jiryang.github.io/img/active_vs_random.png "Random Sampling vs Active Learning"){: width="70%"}{: .aligncenter}


아래 그래프를 보시면 active learning 방식으로 '중요한' 데이터를 선별하여 label을 달면서 학습한 모델과, 그렇지 않고 모든 데이터를 label한 다음 random하게 학습데이터에 추가하면서 (training data 양을 늘려가면서) 학습한 모델의 데이터 양에 따른 정확도를 비교해보실 수 있습니다. 참고로 모든 데이터에 label을 달게 된다면 두 방법 모두 정확도는 이론적으로 같을 것입니다 (Active learning과 반대로 모든 데이터에 label을 달아놓고 모델을 학습하는 것을 _passive learning_ 이라고 부릅니다). 하지만 우리가 특정 성능을 달성하는데 필요한 label cost를 최적화하고자 한다면 active learning이 유리하다는 걸 알 수 있습니다. 점선을 보시면 80% 정확도를 달성하기 위해 active learning 방법을 사용하면 45% 정도의 labeled data만 있으면 되는 반면, passive learning으로는 70% 정도의 labeled data가 필요합니다. 

![Fig2](https://jiryang.github.io/img/model_learning_curve.png "Accuracy per Labeled Data"){: width="70%"}{: .aligncenter}


아래 두 그림은 active learning의 process를 보여줍니다. Active Learning Loop 그림에서는 일부 label된 데이터로 baseline 모델을 학습한 다음, 이를 사용해 '중요한' 데이터를 선별하여 _human oracle_ 을 통해 label을 달고, 이 데이터를 모델에 추가 학습시켜서 모델을 개선하는 과정을 반복해서 수행하면서 모델의 성능을 향상시키는 것을 보실 수 있습니다. Active Learning Dataset 그림에서는 active learning을 거치면서 labeled data가 어떻게 증가하는지를 보여줍니다. Loop를 반복하면서 '중요한' 데이터가 조금씩 추가 label되고, 목표한 accuracy를 달성한 최종 버전에서도 dataset이 partially-labeled된 것을 알 수 있습니다. Label하지 않은 data의 양만큼 비용이 절약되었다고 할 수 있겠죠. 

![Fig3](https://jiryang.github.io/img/active_learning_loop.png "Active Learning Loop"){: width="70%"}{: .aligncenter}

![Fig4](https://jiryang.github.io/img/active_learning_dataset.png "Active Learning Dataset"){: width="70%"}{: .aligncenter}


그럼 학습에 더 도움을 주는 '중요한' 데이터란 어떤 것이고, 어떻게 골라낼 수 있을까요? 

Active learning을 적용할 수 있는 시나리오는 stream-based selective sampling, membership query synthesis와 같은 방법도 있지만 가장 범용적인 pool-based sampling에 대해서만 살펴보도록 하겠습니다. 우선 일부 label된 데이터로 학습하여 '적당한' 성능을 보이는 baseline 모델이 필요합니다. Baseline 모델에 나머지 unlabeled data를 넣어서 inference를 시켜보고, 그 결과를 이용하여 어떤 data에 label을 다는 것이 효과적일지를 정하는 방법입니다. 모델의 입장에서 가장 informative한 데이터는 현재 자신이 판단하기 어려운 헷갈리는 데이터일 것입니다 (위의 첫번째 그림에서 decision boundary에 근접한 데이터들). 이러한 데이터들에 _human oracle_ 이 label을 달아주면 supervised learning을 통해 모델은 기존에 헷갈리던 (inference error가 발생하던) decision boundary를 고쳐나갈 수 있게 됩니다. 그렇기 때문에 inference 결과 uncertainty가 높은 순서대로 데이터를 뽑아 label을 달아주면 됩니다.


Data의 uncertainty를 측정하는 데는 아래와 같이 여러가지 방법들이 있습니다. 데이터, 모델, 태스크 특성에 따라 적절히 사용하시면 될 것 같습니다:

1. Least Confidence

2. Margin Sampling

3. Entropy-Based


Pool-based active learning의 pseudo-code입니다:

> $$\epsilon$$ = training error bound;
> Divide data into unlabelled pool $$P$$ and test set $$S$$;
> Split training pool into batches;
> Randomly select $$k$$ examples from training pool to put in initialized training set $$T$$;
> **while** $$Training Error > \epsilon$$ **do**
>> Train the model using $$T$$;
>> Use the trained model with the test-set, get performance measures;
>> For $$e \in P$$, compute uncertainty for $$e$$;
>> Select $$k$$ most-informative samples based on uncertainty metric;
>> Move these $$k$$ examples to training set;
>> Remove these $$k$$ examples from pool $$P$$;
> **end**

```
this is code block
&#949; = training error bound
(&#949;)
\alpha 
($$\alpha$$)


```

\epsilon = training error bound
\alpha 
$$\alpha$$