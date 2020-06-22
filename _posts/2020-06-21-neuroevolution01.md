---
layout: post
title:  "Deep Neuroevolution"
date:   2020-06-21 12:59:59
categories: Neuroevolution
tags: neuroevolution genetic_algorithm reinforcement_learning uber_ai
excerpt: Deep Neuroevolution과 Deep Learning의 Synergy
mathjax: true
---


코로나 바이러스로 많은 비지니스들이 타격을 입었지만, 사람들의 이동이 줄어들어서 여행산업 쪽이 특히 피해가 컸습니다. 전통적인 항공사나 여행사들 뿐만 아니라 Uber, Airbnb, Lyft와 같은 tech 기업들도 적잖은 타격을 받아서 수천명씩 layoff를 하기에 이르렀죠. 특히 Uber의 경우는 전 세계 고용인원 중 14% 정도에 달하는 3700명 가량을 해고하면서 (Airbnb는 25%인 1900명, Lyft도 20%가 넘는 1200명 이상을 해고했네요), Uber AI의 프로젝트들을 줄여나가겠다고 발표하였습니다 ([링크](https://analyticsindiamag.com/uber-ai-labs-layoffs/)).


Uber AI에서는 메인 비지니스 영역에 필요한 Computer Vision, 자율주행, 음성인식 등과 같은 주제 외에 조금 색다른 포인트의 연구도 해왔는데요, 바로 neuroevolution입니다. Neuroevolution은 Genetic algorithm (GA)과 같은 진화 알고리즘을 사용하여 agent (agent를 구성하는 neural network의 weight나 topology) 또는 rule 같은걸 학습하는 머신러닝 방법으로, artificial life simulation, game playing, evolutionary robotics와 같은 분야에 사용되던 'old school' 기법이라고도 할 수 있습니다. 요새 유명해진 AutoML과 개념적으로 비슷한 neural network의 topological learning 방법인 NEAT ([Neuroevolution of Augmenting Topologies](https://www.cs.ucf.edu/~kstanley/neat.html))를 개발하였던 UCF의 Kenneth O'Stanley 교수님이 Uber AI의 Neuroevolution 분야를 리딩하셨는데요, NEAT는 2000년대 초반에 간단한 방법으로 네트워크의 topology를 효과적으로 진화시키는 결과를 보여주어서 굉장히 인상깊었던 논문입니다.


앞 단락에서 설명드린 neuroevolution의 사용처를 보시면 눈치채셨겠지만 주로 reinforcement learning의 영역이지요? 강화학습 쪽 market requirement도 있고 해서 언젠가 한 번 다루어야겠다 생각을 하고 있었는데요, 이번 기회에 Uber AI에서 진행되었던 neuroevolution 관련 리서치 결과들을 여러 개의 포스팅으로 나누어 훑어보도록 하겠습니다. 중간중간 강화학습 자체에 대한 내용이 적잖이 들어가야 할 것 같은데, 이미 잘 설명된 다른 블로그들이 있는 것 같아서 상당 부분은 인용하도록 하겠습니다.

![Fig1](https://jiryang.github.io/img/neuroevolution.png "Neuroevolution"){: width="80%"}{: .aligncenter}


**Neuroevolution**<br><br>
Neuroevolution은 이름 그대로 neuron을 evolve하는 방식으로 학습을 한다는 의미입니다. Machine learning context에서 이야기를 하는 것일테니 여기서 neuron이란 Artificial Neural Network (ANN)를 뜻합니다. Evolution이란 진화의 방식을 모방한 학습을 이용했다는 의미로, 대부분 population-based optimization 방식인 GA를 사용합니다.<br><br>
_Genetic Algorithm_<br>
GA는 여러 biological evolution theory를 조합해서 만든 머신러닝 알고리즘입니다. Charles Darwin의 'survival of the fittest', Jean-Baptiste Larmark의 'use and disuse inheritance', 그리고 Gregor Mendel의 'crossover & mutation'을 응용하였죠. 동작하는 방식은 다음과 같습니다:
1. Randomly initialized pool of agents로 (stochastic이라 할 만큼) 충분히 큰 population (size N>>)을 생성 (initial generation)
2. 당 generation의 각 agent에게 task를 수행시킴
3. 각 agent의 task에 대한 fitness 값 계산
4. 성능 좋은 n개 (n<<N) agent를 breeding 용으로 선택 (survival of the fittest, natural selection, elitism)
5. 선택된 n개를 mating 시켜 다음 generation의 population을 생성 (crossover & mutation)
6. Stopping criteria (성능, generation 수 등)에 이를 때까지 2$$\sim$$5를 반복
GA는 다음과 같은 process로 동작합니다. Initialization 단계에서 

![Fig2](https://jiryang.github.io/img/ga_process.png "Process of Genetic Algorithm"){: width="50%"}{: .aligncenter}


몇 가지 알아두면 좋은 개념들을 소개합니다:<br>
* Gene<br>
* Chromosome<br>
* Population<br>
* Crossover<br>
* Mutation<br>
* Fitness<br>
* Elitism<br>
* Offspring<br>
* Genotype<br>
* Phenotype<br>


**Deep Neuroevolution**<br><br>


