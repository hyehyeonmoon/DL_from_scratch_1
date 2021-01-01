# [Lec4] Backpropagation

Owners: 혜현 문

# [Lecture4_Backpropagation 요약]

[Summary of cs224n-2019-lecture04-backprop.pdf](%5BLec4%5D%20Backpropagation%202130b2917fdf48c69234ff9e7e984913/Summary_of_cs224n-2019-lecture04-backprop.pdf)

1. 행렬로 표현한 Backpropagation(이전수업리뷰)
2. 개별변수로 표현한 Backpropagation
3. Word2Vec에서 Backpropagation을 이용해 weight을 업데이트시키는 방법
4. Wordvetor를 Training 시켰을 때 일어날 수 있는 성능저하와 Pre-trained word vector와 Training word vector의 선택 상황제시
5. Backpropagation 계산(Single node, Multiple node, 코드)
6. Stuff you should know

-Regularization, Vectorization, Nonlinearities, Initialization, Optimizers, Learning rates

- 2번 참고자료 : Multi Layer Perceptron에서 Backpropagation 공식(서울시립대 "인공지능" 강의_유하진 교수님)
- 5번 참고자료 : Backpropagation 연산자에 따른 gradient 계산방법 요약(서울시립대 "인공지능" 강의_유하진 교수님)

# [Optimizers 추가공부]

[loss optimizers_요약정리.pdf](%5BLec4%5D%20Backpropagation%202130b2917fdf48c69234ff9e7e984913/loss_optimizers_.pdf)

1. Momentum
2. Nesterov Accelerated Gradient(NAG)
3. Adagrad
4. Adadelta
5. Rmsprop
6. Adam

- 자료출처 : 서울서울시립대 "인공지능" 강의_유하진 교수님
- 그 외 다수의 사이트 참고
