# Ch03_Neural network

생성일: 2021년 1월 1일 오후 3:50
태그: Hyehyeon Moon

## [3장 신경망]

신경망은 각 층의 뉴런들이 다음 층의 뉴런으로 신호를 전달한다는 점에서 앞 장의 퍼셉트론과 같습니다. 하지만 다음 뉴런으로 갈 때 신호를 변화시키는 활성화 함수에 차이가 있었습니다. (시모이드 함수, ReLU 함수)

행렬 연산이 신경망 구현을 편리하게 만드는 것을 확인하고, 단일 표본에 대해, 한 단계 더 나아가 batch size의 표본에 대해 신경망을 구현해 보았습니다. 출력층의 활성화 함수로 회귀에서는 항등 함수(identity function), 분류에서는 소프트맥스 함수(softmax function)를 주로 이용합니다.

실습으로 MNIST 데이터셋을 이용한 손글씨 숫자를 인식해 보았습니다.

## [정리]

- 신경망에서는 활성화 함수로 시그모이드 함수와 ReLU 함수 같은 매끄럽게 변화하는 함수를 이용한다.
- 넘파이의 다차원 배열을 잘 사용하면 신경망을 효율적으로 구현할 수 있다.
- 기계학습 문제는 크게 회귀와 분류로 나눌 수 있다.
- 출력층의 활성화 함수로는 회귀에서는 주로 항등 함수를, 분류에서는 주로 소프트맥스 함수를 이용한다.
- 분류에서는 출력층의 뉴런 수를 분류하려는 클래스 수와 같게 설정한다.
- 입력 데이터를 묶은 것을 배치라 하며, 추론 처리를 이 배치 단위로 진행하면 결과를 훨씬 빠르게 얻을 수 있다.

## [파일 설명]

step_function : 계단 함수를 구현한 코드입니다.

sigmoid : 시그모이드 함수를 구현한 코드입니다.

relu : ReLU 함수를 구현한 코드입니다.

neural net : 3층 신경망을 구현한 코드입니다.

mnist_show : MNIST 데이터셋을 읽어와 훈련 데이터 중 0번째 이미지를 화면에 출력합니다.

mnist_single sample : 신경망으로 손글씨 숫자 그림을 추론합니다.

neuralnet_mnist_batch : mnist_single sample에 배치 처리 기능을 더했습니다.

sample_weight.pkl : 미리 학습해둔 가종치 매개변수의 값들입니다.

.ipynb 파일의 오류를 대비해 .py 파일을 따로 폴더에 저장해 놓았습니다.

## [심화]

### 활성화 함수의 종류

- 선형 함수 : f(net)=net
- 비선형 함수 : step function, sigmoid, tanh, ReLU,  Leaky ReLU, PReLU, ELU, Maxout,

![Ch03_Neural%20network%204e3d741244194c6cabba4ccb613325c1/Untitled.png](Ch03_Neural%20network%204e3d741244194c6cabba4ccb613325c1/Untitled.png)

### 1. 활성화 함수로 선형 함수를 사용하지 않는 이유

은닉층의 효과를 얻을 수 없기 때문입니다.

### 2. Sigmoid의 단점

![Ch03_Neural%20network%204e3d741244194c6cabba4ccb613325c1/Untitled%201.png](Ch03_Neural%20network%204e3d741244194c6cabba4ccb613325c1/Untitled%201.png)

- Gradient Vanishing 현상이 발생

미분함수에 대해 x=0에서 최대값 1/4을 가지고, |x|가 커질수록 0에 수렴하기 때문에 Backpropagation 시 gradient vanishing이 발생합니다.

- 함수값 중심이 0이 아니므로 학습이 느려짐

만약 모든 x값들이 같은 부호(ex. for all x is positive) 라고 가정하고 아래의 파라미터 w에 대한 미분함수식을 살펴봅니다.

$$\frac{∂L}{∂W}=\frac{∂L}{∂a} \times \frac{∂a}{∂W} \\ \frac{∂a}{∂W}=x \\ \frac{∂L}{∂W}=\frac{∂L}{∂a} \times x $$

![Ch03_Neural%20network%204e3d741244194c6cabba4ccb613325c1/Untitled%202.png](Ch03_Neural%20network%204e3d741244194c6cabba4ccb613325c1/Untitled%202.png)

위 식에서 모든 x가 양수라면 결국 ∂L/∂w는 ∂L/∂a 부호에 의해 결정됩니다. 따라서 한 노드에 대해 모든 파라미터w의 미분값은 모두 같은 부호를 같게됩니다. 따라서 같은 방향으로 update되는데 이러한 과정은 학습을 zigzag 형태로 만들어 느리게 만드는 원인이 됩니다.

- exp 함수 사용시 비용이 큼

### 3. tanh(Hyperbolic tangent function)의 단점

![Ch03_Neural%20network%204e3d741244194c6cabba4ccb613325c1/Untitled%203.png](Ch03_Neural%20network%204e3d741244194c6cabba4ccb613325c1/Untitled%203.png)

- tanh 함수는 함수의 중심값을 0으로 옮겨 sigmoid의 최적화 과정이 느려지는 문제를 해결합니다.
- 하지만, gradient vanishing 문제는 여전히 남아있습니다.

### 4. ReLU

![Ch03_Neural%20network%204e3d741244194c6cabba4ccb613325c1/Untitled%204.png](Ch03_Neural%20network%204e3d741244194c6cabba4ccb613325c1/Untitled%204.png)

- 연산비용이 크지 않고, 구현이 매우 간단함→sigmoid, tanh 함수와 비교시 학습이 빨라집니다.
- x<0인 값들에 대해서는 기울기가 0이기 때문에 뉴런이 죽을 수 있는 단점이 존재합니다.

### 5. Leaky ReLU

![Ch03_Neural%20network%204e3d741244194c6cabba4ccb613325c1/Untitled%205.png](Ch03_Neural%20network%204e3d741244194c6cabba4ccb613325c1/Untitled%205.png)

- ReLU의 뉴런이 죽는 현상을 해결하기 위해 음수의 x값에 대해 0.01을 곱해주어 음수 x값의 미분값이 0이 되지 않게 만들어줍니다.

### 6. PReLU

![Ch03_Neural%20network%204e3d741244194c6cabba4ccb613325c1/Untitled%206.png](Ch03_Neural%20network%204e3d741244194c6cabba4ccb613325c1/Untitled%206.png)

- Leakly ReLU와 거의 유사하지만 새로운 파라미터 α 를 추가하여 x<0에서 기울기를 학습할 수 있게 합니다.

### 7. ELU(Exponential Linear Unit)

![Ch03_Neural%20network%204e3d741244194c6cabba4ccb613325c1/Untitled%207.png](Ch03_Neural%20network%204e3d741244194c6cabba4ccb613325c1/Untitled%207.png)

- ReLU의 모든 장점을 포함하면서 뉴런이 죽는 문제를 해결합니다.
- 일반적인 ReLU와 달리 exp함수를 계산하는 비용이 발생합니다.

### 8. Maxout 함수

![Ch03_Neural%20network%204e3d741244194c6cabba4ccb613325c1/Untitled%208.png](Ch03_Neural%20network%204e3d741244194c6cabba4ccb613325c1/Untitled%208.png)

- ReLU가 가지는 모든 장점을 가졌으며, dying ReLU문제 또한 해결합니다.
- 계산량이 복잡하다는 단점이 있습니다.

### 9. 결론

- 가장 많이 사용되는 함수는 ReLU입니다.
- ReLU를 사용한 이후 Leakly ReLU등 ReLU계열의 다른 함수도 사용 해봅니다.
- sigmoid의 경우에는 사용하지 않도록 합니다.
- tanh의 경우도 큰 성능은 나오지 않습니다.
- 2018년의 글을 참고하였으므로 현재는 다를 수 있음을 염두에 두어야 합니다.

### 참고 사이트 및 이미지 출처

[딥러닝에서 사용하는 활성화함수](https://reniew.github.io/12/)

[8. activation function - saturation현상 / zigzag현상 / ReLU의 등장](https://nittaku.tistory.com/267)

[ML - Sigmoid 대신 ReLU? 상황에 맞는 활성화 함수 사용하기](https://medium.com/@kmkgabia/ml-sigmoid-%EB%8C%80%EC%8B%A0-relu-%EC%83%81%ED%99%A9%EC%97%90-%EB%A7%9E%EB%8A%94-%ED%99%9C%EC%84%B1%ED%99%94-%ED%95%A8%EC%88%98-%EC%82%AC%EC%9A%A9%ED%95%98%EA%B8%B0-c65f620ad6fd)