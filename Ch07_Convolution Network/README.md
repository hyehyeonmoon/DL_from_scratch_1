## [7장 합성곱 신경망(CNN)]

합성곱 연산을 배우고 합성곱 계층을 구현해 본다.

풀링 계층의 연상방법을 배우고 풀링 계층을 구현해 본다.

합성곱 계층, 풀링 계층, 완전연결 계층을 사용하여 CNN을 구현하고, MNIST data로 학습과 평가를 해본다.

CNN은 계층이 깊어질수록 추상화적이고 고급정보가 추출된다.

CNN의 대표적인 모형으로 LeNet과 AlexNet이 있다.

## [정리]

- CNN은 지금까지의 완전연결 계층 네트워크에 합성곱 계층과 풀링 계층을 새로 추가한다.
- 합성곱 계층과 풀링 계층은 im2col (이미지를 행렬로 전개하는 함수)을 이용하면 간단하고 효율적으로 구현할 수 있다.
- CNN을 시각화해보면 계층이 깊어질수록 고급 정보가 추출되는 모습을 확인할 수 있다.
- 대표적인 CNN에는 LeNet과 AlexNet이 있다.
- 딥러닝의 발전에는 빅 데이터와 GPU가 크게 기여했다.

## [파일 설명]

![KakaoTalk_20210119_000212051](https://user-images.githubusercontent.com/55529617/104999570-6b6bdc80-5a70-11eb-9610-3cce81d4af6c.jpg)

CNN의 구현모델을 위의 사진과 같이 체계적으로 짤 수 있었다.

|File |Description |
|:-- |:-- |
| Conv_forward | im2col, convnet, pooling의 forward가 구현된 함수를 포함|
| Conv_backward| im2col, col2im, convnet, pooling의 forward, backward가 구현된 utils, model에 해당하는 파일|
| SimpleConvNet| “Convolution-ReLU-Pooling-Affine-ReLU-Affine-Softmax” 순으로 흐르는 단순한 합성곱 신경망(CNN)을 전체적으로 구현한 파일|
| Trainer| model(Conv_backward) 부분과 train(SimpleConvNet)을 합쳐 실질적인 학습을 시키는 파일|
|predict_convnet | Trainer를 이용해 MNIST 데이터를 학습 및 평가하는 파일|

그 외 "밑바닥부터 시작하는 딥러닝1"에서 CNN 공부에 사용한 파일이다.

|File |Description |
|:-- |:-- |
|Detail_matrix| “Convolution-ReLU-Pooling-Affine-ReLU-Affine-Softmax” 각 계층에서 행렬의 변화를 자세히 살펴봄 |
|apply_filter| 학습된 필터를 이용해 cactus_gray.png 파일에 적용 |
|gradient_check| SimpleConvNet의 수치미분과 역전파미분 비교를 통해 기울기를 올바르게 계산하는지 확인|
|visualize_filter| 합성곱 1번째 층의 가중치를 학습 전과 후로 나눠 시각화|
|params.pkl| 미리 학습된 가중치 값들|


