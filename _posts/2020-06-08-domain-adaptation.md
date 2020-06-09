---
layout: post
title:  "Domain Adaptation: Learning to Learn"
date:   2020-06-08 00:00:01
categories: GeneralML
tags: domain_adaptation transfer_learning
excerpt: 도메인 적응 리뷰 및 예시
mathjax: true
---


Transfer learning이라는 말을 여기저기서 들어보셨을 것입니다. 하지만 'pre-trained model로 신규 학습을 시작하는 것' 이라고 알고 계시는 분들도 많을 것 같습니다. 보통 image classifier를 만든다면 학습 데이터를 구하고, 네트워크를 정하고, 이제 할 일이 이 네트워크를 ImageNet으로 학습시킨 pre-trained model을 찾는 것이죠. Pre-trained model을 사용하면 내 데이터에 대한 classifier 학습이 빨리 되도록 도와주기도 하고, 특히 내 데이터가 충분히 많지 않아서 네트워크를 train from scratch하기에 부족하다면 pre-trained model을 찾느냐 못찾느냐는 classifier의 성패를 좌우하게 되겠죠.


내 dataset과는 생김새도, class 갯수도 다른 ImageNet으로 학습한 pre-trained model이 왜 학습에 도움이 되는지? 이런 'transfer learning'은 어떤 경우에나 사용할 수 있는 만능인지? Transfer learning이 도대체 뭔지? 듣고보니 궁금해지는 질문들이죠?


Real-world example을 이용해서 좀 다른 방향으로 이야기를 해보죠.<br>
우리나라의 인공위성 중 아리랑 3호는 약 하루 2회 (주간, 야간) 한반도 상공을 지나며 고해상도 전자광학카메라로 사진을 촬영해왔습니다. 기상, 군사 등 다양한 목적을 가지고 다량의 사진을 찍었을 텐데요, 70cm급 해상도를 가지고 있다고 하니 촬영된 물체가 무엇인지 육안으로도 어느정도는 확인이 가능할 것 같긴 합니다만, 그래도 명확치 않은 것들도 있고 워낙 사진의 양이 많으니 object recognizer를 만들어서 자동으로 분류시키기로 하였습니다. 비용과 시간을 들여서 detect하고자 하는 object들에 대해 수많은 optical image들에 대한 annotation 작업을 마쳤습니다. 이제 위성에서 사진을 보내오면 자동으로 물체를 인식하여서 비구름이 어떻게 움직이는지 알아 강수를 예측하고, 북한의 어떤 함정이 몇 대 어느 항구에 정박해 있는지 알아 자동으로 국군의 준비태세를 갖추게 되었습니다.


그런데 optical camera에는 커다란 단점이 있었는데요, 그 이름 그대로 optical하다는 점입니다. 빛이 없거나 가리면 촬영할 수가 없다는 뜻이지요. 하루에 2회 한반도 상공을 촬영할 수 있는데 구름이 끼어 있으면 지상 촬영이 안된다거나, 야간의 경우 촬영한 이미지로 지상의 object 관찰이 어렵다는 취약점이 있었습니다. 이러한 단점을 극복하고자 위성에서 지상에 레이다파를 쏘고, 다양한 굴곡면에 반사되어 나온 레이다파의 시차를 이용하여 해당 굴곡면을 가진 object의 형태를 파악하는 합성개구레이다 (Synthetic Aperture Radar, SAR) 기술을 개발, 차기 위성인 아리랑 5호와 아리랑 3A에 탑재하였습니다. 더이상 빛의 유무와 occlusion에 제약이 없어져서 밤에도, 구름낀 날에도 지상의 object 관찰이 가능해졌습니다. 그런데 바뀐 입력 이미지에서 object recognition을 돌리려니 SAR 이미지로 모델을 새로 학습해야 합니다. 여기서 문제가 발생합니다. 사람의 눈과 동일한 방식으로 동작한Optical image는 우리의 눈과 같은 방식으로 동작하기 때문에 그 결과물도 우리가 인식하는 것과 동일한 반면, SAR은 그렇지가 않다는 점입니다. 기계적으로야 가능하겠지만 SAR 촬영 object를 눈으로 구분하는 것이 쉽지 않습니다. 아래는 optical과 SAR로 촬영한 각종 전술차량의 그림입니다. 눈으로는 식별이 가능하지 않지요.

![Fig1](https://jiryang.github.io/img/tank_optical_vs_sar.PNG "Optical and SAR Sample Images"){: width="70%"}{: .aligncenter}


모델 재학습을 하기로 결정하였습니다. 학습 데이터를 모으고 annotation을 달면 되는데... 극소수의 전문가 외에는 SAR 이미지를 식별조차 할 수 없어서 annotation을 달기가 어렵습니다. 예상 비용은 작업 시간은 무한정 늘어만 갑니다. [Active learning](https://jiryang.github.io/2020/05/31/data-labeling/)이든 뭐든 비용과 시간만 아낄 수 있다면 어떤 방법이라도 좋습니다. 마음 한 켠에는 아리랑 3호의 작년에 거금을 들여 optical image로 annotation을 달아 만들어두었던 학습 데이터가 아까와 죽겠습니다. 이걸 어떻게 써먹을 방법은 없는걸까요...


있습니다. 서론이 길었습니다만 오늘의 주제인 domain adaptation이 이런 경우의 문제를 해결해 줄 수 있는 한 방법입니다. 우선 문제를 formalize 하는 것부터 설명을 시작하겠습니다. 그러면서 transfer learning과 domain adaptation이 어떤 차이가 있는지도 이야기를 해보도록 하겠습니다.


문제가 정의되는, 혹은 데이터가 정의되는 도메인 $$\mathcal{D}$$는 $$\mathcal{d}$$ 차원을 가지는 데이터 $$\mathcal{X}$$와 그 확률분포 $$P(\mathcal{X})$$로써 다음과 같이 정의됩니다:<br>
$$\mathcal{D}=\{\mathcal{X}, P(\mathcal{X})\}$$


$$\mathcal{X}$$의 특정 set인 $$X={x_1, x_2, ..., x_n}\in\mathcal{X}$$의 label을 $$Y={y_1, y_2, ..., y_n}\in\mathcal{Y}$$라고 할 때, task $$\mathcal{T}$$를 입력 $$X$$가 $$Y$$의 확률을 가질 경우를 나타내는 조건부 확률인 $$P(Y \mid X)$$ 라고 정의할 수 있습니다.


도메인 적응(Domain Adaptation, DA)은 task(예를 들면 앞에서의 object recognition)의 domain이 모델을 학습했던 것(source domain, 예를 들면 optimal image domain)에서 어느정도 관련은 있지만 동일하지는 않은 다른 도메인(target domain, 예를 들면 SAR image domain)으로 변경되었을 때, source domain에서 학습된 knowledge를 transfer 해 주어 target domain에서 이용할 수 있도록 해줍니다. 계속해서 formalize 보시죠.


앞에서의 domain과 task 정의를 이용하여 source와 target의 domain과 task를 다음과 같이 표현할 수 있습니다:<br>
Source domain: $$\mathcal{D^S}=\{\mathcal{X^S}, P(\mathcal{X^S})\}$$, source task: $$\mathcal{T^S}=\{\mathcal{Y^S}, P(Y^S \mid X^S)\}$$<br>
Target domain: $$\mathcal{D^T}=\{\mathcal{X^T}, P(\mathcal{X^T})\}$$, target task: $$\mathcal{T^T}=\{\mathcal{Y^T}, P(Y^T \mid X^T)\}$$


우리가 일반적으로 모델을 학습시키는 경우에는 training과 test 사이에 task도 변하지 않고, 하나의 domain에 속한 데이터셋을 training/test 셋으로 나누어 사용하기 때문에 $$\mathcal{D^S}=\mathcal{D^T}$$, $$\mathcal{T^S}=\mathcal{T^T}$$ 라고 할 수 있겠습니다. 그럼 위에서 얘기한 optical vs SAR image data의 경우는 어떨까요? 이 경우는 object recognition이라는 task는 변하지 않았지만 data의 domain은 optical에서 SAR로 바뀌었다고 볼 수 있기 때문에 $$\mathcal{D^S} \neq \mathcal{D^T}$$, $$\mathcal{T^S}=\mathcal{T^T}$$ 라고 할 수 있겠습니다. 이런 식으로 $$\mathcal{D^S}=\mathcal{D^T}$$, $$\mathcal{T^S} \neq \mathcal{T^T}$$ 인 경우도 있겠지요? 각각의 경우를 좀 더 general하게 구분해보겠습니다.

**1. Same domain, same task** <br>
$$\mathcal{D^S}=\mathcal{D^T}$$, $$\mathcal{T^S}=\mathcal{T^T}$$<br>
앞서 설명한대로 일반적인 ML 문제의 경우입니다. 주어진 데이터를 training과 test로 나눠서 학습하고 infer하면 됩니다.



**2. Different domain, different task** <br>
$$\mathcal{D^S} \neq \mathcal{D^T}$$, $$\mathcal{T^S} \neq \mathcal{T^T}$$<br>
아.. 생각만 해도 제일 골치아픈 경우입니다. 이런 문제를 과연 풀 수 있을까요.. Source와 target domain이 완전히 다른 경우는 knowledge transfer를 한다는 것 자체가 말이 안되고, 다르긴 하지만 어느정도 유사성이 있어야 합니다. Task의 경우도 마찬가지입니다. 구분을 하자면 _inductive transfer learning_ 이나 _unsupervised transfer learning_ 에 속하는 문제들이겠지만, 이런 경우는 그냥 새로운 데이터로 재학습하는게 나을 것 같습니다. Self-taught learning 이라는 기법도 있지만 toy example에서 6-70% 정도 성능을 보였고, 이후 지속적으로 개선되지 않은 것 같습니다 (제가 모르는 것일 수도 있음). 여튼 이번에 다루고자 하는 topic이 아니므로 이정도에서 패쓰합니다.


**3. Same domain, different task** <br>
$$\mathcal{D^S}=\mathcal{D^T}$$, $$\mathcal{T^S} \neq \mathcal{T^T}$$<br>
Source domain의 데이터가 충분하고, target domain의 labeled 데이터가 어느정도 있다면 source domain에서 학습된 모델이 source domain 상에서의 성능을 유지하면서 target domain에서도 동작하도록 학습을 할 수 있습니다. ImageNet pre-trained 모델로 face recognition에 적용하는게 이런 경우의 예라고도 할 수 있겠습니다. _Inductive transfer learning_ 또는 _multi-task learning_ 정도로 구분지을 수 있겠네요. 이 부분도 일단 out of topic입니다.


**4. Different domain, same task** <br>
$$\mathcal{D^S}=\mathcal{D^T}$$, $$\mathcal{T^S} \neq \mathcal{T^T}$$<br>
이게 바로 오늘 이야기를 하고자 하는 DA 입니다. 도메인을 다음과 같이 정의하였죠: $$\mathcal{D}=\{\mathcal{X}, P(\mathcal{X})\}$$. DA를 좀 더 세분하자면 data source 자체가 다른, 그러니까 $$\mathcal{X^S} \neq \mathcal{X^T}$$ 때문에 $$\mathcal{D^S} \neq \mathcal{D^T}$$가 되는 경우를 _heterogeneous DA_ 라고 하고, data source 자체는 같지만 ($$\mathcal{X^S} = \mathcal{X^T}$$) 그 분포가 달라서 ($$P(\mathcal{X^S}) \neq P(\mathcal{X^T})$$) $$\mathcal{D^S} \neq \mathcal{D^T}$$가 되는 경우를 _homogeneous DA_ 라고 합니다. Optical vs SAR object recognition는 식별하고자 하는 object는 동일하나, 촬영 기법의 변화로 인해 해당 object를 나타내는 데이터(이미지)의 분포가 달라진 경우이기 때문에 _homogeneous DA_ 로 구분됩니다. 특히 후자에 집중해서 볼 예정인데요, 실제 field에서 이러한 요구사항이 적지 않을 것 같기 때문입니다. Optical vs SAR의 경우와 같이 측정 장비의 업그레이드로 인해 task는 동일하나 예전 모델과 데이터 domain이 달라져버리는 경우가 종종 있지 싶습니다. 예를 들면 MRI 장비를 가지고 있던 병원에서 CT 기계를 들여놓는 경우도 마찬가지이죠. MRI 종양 검출 모델을 학습시켜놓았는데 CT로 바꾸고 나서 쌓여있던 데이터와 모델을 버리기엔 아깝겠죠. 여러 논문들에 나온 experiment들에는 webcam 이미지와 DSLR 이미지를 사용한 object recognition, 실제와 합성 이미지를 사용한 드론 detection, 하나의 빅데이터의 object annotation을 이용해 다른 빅데이터 object annotation 달기, USPS 숫자데이터와 MNIST를 이용한 손글씨 숫자인식 등 다양한 문제들을 다루고 있습니다.


(일반적으로 말하는 **Transfer learning (전이학습)** 이란 위의 2, 3, 4번, 그러니깐 한 도메인에서 학습된 knowledge를 다른 도메인에 적용하거나, 한 task에 대해 학습한 knowledge를 다른 task에 적용하거나, 혹은 둘 다 동시에 하는 경우를 모두 일컫는 말입니다.)



논문링크 (survey 논문 위주 리스트업):
[A Survey on Transfer Learning](https://www.cse.ust.hk/~qyang/Docs/2009/tkde_transfer_learning.pdf)
[Domain Adaptation for Visual Applications: A Comprehensive Survey](https://arxiv.org/pdf/1702.05374.pdf)
[Deep Visual Domain Adaptation: A Survey](https://arxiv.org/pdf/1802.03601.pdf)
[Recent Advances in Transfer Learning for Cross-Dataset Visual Recognition: A Problem-Oriented Perspective](https://arxiv.org/pdf/1705.04396.pdf)