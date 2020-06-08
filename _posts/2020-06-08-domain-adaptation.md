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
$$\mathcal{D}={\mathcal{X}, P(\mathcal{X})}$$


$$\mathcal{X}$$의 특정 set인 $$X={x_1, x_2, ..., x_n}\in\mathcal{X}$$의 label을 $$Y={y_1, y_2, ..., y_n}\in\mathcal{Y}$$라고 할 때, task $$\mathcal{T}$$를 입력 $$X$$가 $$Y$$의 확률을 가질 경우를 나타내는 조건부 확률인 $$P(Y|X)$$ 라고 정의할 수 있습니다.



