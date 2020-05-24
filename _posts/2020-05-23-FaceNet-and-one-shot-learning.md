---
layout: post
title:  "FaceNet, One-Shot Learning, Triplet Loss"
date:   2020-05-22 22:59:59
categories: Deepfake
tags: facenet one-shot_learning triplet_loss
excerpt: FaceNet의 Triplelet One-Shot Learning
mathjax: true
---

> 총 n명의 국제 테러리스트 명단이 공개되었습니다. 공항 검색대에 설치된 CCTV로 N명 (N>>n) 입국자들의 얼굴을 찍어서 테러리스트인지 판정하는 모델을 만드십시오.

이런 태스크가 떨어지면 어떤 모델을 만들까요? 그냥 단순한 classifier를 붙이면 될까요? 안될까요?
Classifier로 만들면 예상되는 문제들은 어떤게 있을까요?

먼저 테러리스트 숫자인 n보다 훨씬 큰 N명의 사람들은 어느 클래스로 보내죠? 클래스 갯수를 n+1로 만들어서 non-terrorist를 하나로 몰아넣는다? 이 클래스에만 데이터가 넘쳐나겠죠? 뭐 그럼에도 불구하고 학습데이터를 실제 비율이랑 비슷하게 만들어서 열심히 학습을 시켜볼 수는 있겠습니다만 문제가 또 발생합니다. 테러리스트 데이터베이스가 업데이트되어서 명단이 추가되었거든요.

이젠 어떻게 해야할까요? FC layer에서 class 숫자를 하나 더 늘리고, 예전 데이터 일부에 추가된 테러리스트 얼굴을 더해서 추가학습을 시켜볼까요? 어느정도는 성능이 나올 수는 있지만 보장이 가능할까요? 데이터베이스가 계속 업데이트가 되면 이 때마다 매번 추가학습을 할 수 있을까요? 학습이 진행되는 동안 테러리스트가 공항에 도착하면 어떻게 하나요?

FaceNet은 이런 문제들을 해결하기 위한 방법론을 제시합니다. 비단 face domain에서 뿐만 아니라, use-case만 맞는다면 다양한 classification-based 솔루션에 적용할 수 있는 솔루션을요. 간단히 말씀드리면 big dataset의 low-dimensional embedding을, 동일한 class를 가지는 data들끼리는 distance가 가깝도록, 다른 class의 data들끼리는 distance가 멀도록 학습해서, 두 data가 같은 클래스인지 아닌지를 이 embedding 안에서의 distance로 구별해내는 generalizable 모델을 만든다는 것입니다. 테러리스트의 예로 이야기하자면, FaceNet은 일반적으로 두 사진의 얼굴이 동일 인물인지 아닌지를 학습하기 때문에 새로운 테러리스트가 추가되더라도 CCTV에 찍힌 인물이 테러리스트 명단에 있는지 없는지는 추가학습 없이 one-shot으로 n+$\alpha$개의 이미지랑만 비교해보면 된다는 것이죠. 그것도 low-dimensional embedding을 사용하기 때문에 빠르게 가능합니다.

Low-dimensional embedding은 일반 DNN classifier도 사용하는데요? 심지어 PCA를 사용한 eigenface도 low-dimensional embedding을 쓰는건데 무슨 차이가 있을까요? 저자는 Triplet Loss라는 개념을 고안해서 이 문제를 해결하였습니다.



이 Triple Loss는 동일한 구조에 weight까지 공유하는 3개의 동일한 네트워크를 사용하여, 이런걸 샴 네트워크(Siamese network)라고 하죠, 병렬적으로 빠르게 계산이 가능합니다. 하지만 학습 데이터의 triplet을 구성하는 것에 문제가 생깁니다. 예를 들어 학습 데이터에 10,000명의 face ID가 있고 각 ID마다 30장 씩의 사진이 들어있다고 하면, 조합 가능한 triplet의 경우의 수가 $$(30\times10,000)_{anchor}\times29_{positive}\times(30\times9,999)_{negative}$$나 될 것입니다 (계산 맞나요 ㅠㅠ, 암튼 엄청 커집니다).