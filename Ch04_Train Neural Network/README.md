# Ch_04 Train Neural Network

생성일: Jan 1, 2021 3:50 PM
태그: 상현 김

**[4장 신경망 학습]**

학습이란 훈련 데이터로부터 가중치 매개변수의 최적값을 자동으로 획득하는 것입니다. 신경망 학습의 현재 상태를 나타내는 지표로 손실 함수를 이용합니다. 주로 사용하는 손실 함수의 종류로는 평균 제곱 오차(MSE)와 교차 엔트로피 오차(CE)를 사용합니다

손실함수를 최소로 하는 매개변수를 찾기 위해 미분을 사용합니다.  수치 미분을 통해 얻은 기울기(gradient)를 이용해 경사 하강법으로 최솟값을 탐색합니다

위에 언급한 손실함수들과 수치미분 및 경사 하강법을 구현했습니다. 나아가 2층 신경망을 구현한 후 앞서 구현한 손실함수와 경사 하강법을 이용해 MNIST데이터를 이용하여 신경망 학습을 구현했습니다.

**[정리]**

- 기계학습에서 사용하는 데이터셋은 훈련 데이터와 시험 데이터로 나눠 사용한다.
- 훈련 데이터에서 학습한 모델의 범용 능력을 시험 데이터로 평가한다.
- 신경망 학습은 손실 함수를 지표로, 손실 함수의 값이 작아지는 방향으로 가중치 매개변수를 갱신한다.
- 가중치 매개변수를 갱신할 때는 가중치 매개변수의 기울기를 이용하고, 기울어진 방향으로 가중치의 값을 갱신하는 작업을 반복한다.

- 아주 작은 값을 주었을 때의 차분으로 미분을 구하는 것을 수치 미분이라고 한다.
- 수치 미분을 이용해 가중치 매개변수의 기울기를 구할 수 있다.
- 수치 미분을 이용한 계산에는 시간이 걸리지만, 그 구현은 간단하다. 한편, 다음 장에서 구현하는 다소 복잡한 오차역전파법은 기울기를 고속으로 구할 수 있다.

**[파일 설명]**

loss_function: mean square error, cross entropy error를 구현한 코드입니다

gradient: 수치 미분과 경사 하강법을 구현한 코드입니다

simpleNet: 간단한 신경망을 구현한 코드입니다

two_layer_net: 2층 신경망을 구현한 코드입니다

train_neuralnet: 2층 신경망을 이용해 신경망 학습을 구현한 코드입니다

[심화]

손실함수로 Mean Square Error와 Cross Entropy Error를 사용하는 이유와 차이점

$MSE = \frac 1 2 \Sigma_{k} (y_{k},t_{k})^2$

$$CEE = -\Sigma_{k} t_{k}log(y_{k})$$

1. 오차역전파법(Backpropagation) 관점

    $$L(f_{\theta}(x),y) = \Sigma_{i}L(f_{\theta}(x_{i}),y_{i})$$

- 오차역전파법의 가정을 만족하는 함수
    - 가정1: Total loss of DNN over training samples is the sum of loss for each training sample
    - 가정2: Loss for each training example is a function of final output of DNN
- 오차 역전파법의 관점에서 MSE와 CEE의 차이점
    - MSE의 경우 마지막 활성화 함수의 미분값이 살아있는 반면 CEE의 경우 마지막 활성화 함수의 미분값이 계산에 의해 상쇄됩니다. 따라서 MSE가 CEE보다 gradient vanishing problem에 조금 더 취약합니다.

2. 최대 가능도(Maximum Likelihood) 관점

$$p(y|f_{\theta}(x))$$

네트워크 출력값이 주어졌을 때, y가 나올 확률이 최대가 되는 가중치를 찾는 방법입니다.

→ 해당 확률분포가 어떤 확률분포를 따를지 가정해야합니다. ex) 가우시안 분포, 베르누이 분포

![70583D1E-8213-43EC-B4F8-7268BB555E8B](https://user-images.githubusercontent.com/68596881/103986597-63897e00-51ce-11eb-8fd9-c3ed3071308d.jpeg)

가우시안 분포를 따를때의 예시

$$\theta^{*} = argmin_{\theta}[-log(p(y|f_{\theta}(x)))]$$

오차역전파의 가정을 만족시키기 위해 log-likelihood를 사용합니다.

- i.i.d 조건하에 곱으로 표현되는 것을 log를 사용하여 합으로 표현하는 방법으로 바꿔 오차역전파의 가정을 만족시킵니다.

$$-log(p(y|f_{\theta}(x))) = -\Sigma_{i} log(p(y_{i}|f_{\theta}(x_{i})))$$

![21F4F3A4-EB42-4BD6-87FD-4E9185B6CF76](https://user-images.githubusercontent.com/68596881/103986765-a77c8300-51ce-11eb-8d7b-d7aa6c7968a7.jpeg)

일변량 분포일때 식 전개 → 다변량 분포에서도 비슷한 식으로 전개가능

따라서, 신경망을 통해 Likelihood의 파라미터값들을 예측하는 것이다.

정리

- 주어진 조건에서 y의 확률분포가 연속형(가우시안)일 경우: MSE → regression 문제에서 사용
- 주어진 조건에서 y의 확률분포가 이산형(베르누이)일 경우: CEE → classification 문제에서 사용

이외의 몇 가지 손실함수들

- L1loss: 평균 절대값 오차
- PoissonNLLloss: 포아송 분포를 따르는 타겟의 네거티브 로그 가능도
- Hinge, soft margin loss: SVM의 원리를 이용한 손실함수들
- KLDivloss: KL Divergence를 사용한 손실함수

이외에도 여러 가지 손실함수가 존재하며 위의 손실함수들은 pytorch에서 구현된 일부 손실함수를 예시로 가지고 왔습니다.

참고 사이트

[오토인코더의 모든 것 - 1/3](https://www.youtube.com/watch?v=o_peo6U7IRM)

위 심화의 내용, 그래프, 수식은 모두 "오토인코더의 모든 것 강의 1편"을 참고했습니다.

[[딥러닝] 목적/손실 함수(Loss Function) 이해 및 종류](https://needjarvis.tistory.com/567)

[torch.nn - PyTorch 1.7.0 documentation](https://pytorch.org/docs/stable/nn.html#loss-functions)
