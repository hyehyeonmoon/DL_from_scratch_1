# Ch_05 BackPropagation

**[5장 오차역전파(Backpropagation)]**


가중치와 매개변수의 기울기를 효율적으로 계산하는 '오차역전파'법을 배웠다. 
수치미분을 통하여 신경망을 갱신하려면 미분과정에서 미세한 delta값을 더한 순전파를 계산하고 손실함수를 계산하여 각 노드들의 W,b의 값들의 기울기를 구한 편미분을 실행해야해서 연산량이 많다.
<br>
<br>
하지만, 역전파는 연쇄법칙을 원리로 하여 각 노드별 함수를 국소적 계산을 통해서 단순하고 빠르게 차례대로 계산할 수 있다. <br>

**[각 노드 오차역전파법]**

- 덧셈노드 : 그대로 흘려보냄

- 곱셈노드 : 입력신호를 서로 바꾼 값으로 곱해서
- relu: x>0일때 1을 전파 / x <=0 일떄, 0 전파
- sigmoid: 출력값 y만으로 역전파 계산 가능
- Affine : input값과 weight값들을 행렬 곱하여 계산하고 편향을 더하여 역전파 계산 
- softmax with loss: 역전파를 통해서 계산할때 softmax-crossentropy를 이용하여 결과값을 구할 경우  y(신경망이 예측한 정답 확률)-t(실제 정답)으로 말끔히 떨어짐

**[정리]**

- 계산 그래프를 이용하면 계산 과정을 시각적으로 파악할 수 있다.
- 계산 그래프의 노드는 국소적 계산으로 구성된다. 국소적 계산을 조합해 전체 계산을 구성한다.
- 계산 그래프의 순전파는 통상의 계산을 수행한다. 한편, 계산 그래프의 역전파로는 각 노드의 미분을 구할 수 있다.
- 신경망의 구성 요소를 계층으로 구현하여 기울기를 효율적으로 계산할 수 있다(오차역전파법).
- 수치 미분과 오차역전파법의 결과를 비교하면 오차역전파법의 구현에 잘못이 없는지 확인할 수 있다(기울기 확인).

**[파일 설명]**
- buy_apple.py : 사과 2개를 구입하는 예제의 순전파와 역전파 구현
- buy_apple_orange.py : 사과와 오랜지를 구입하는 예제의 순전파와 역전파 구현
- gradient_check.py : 수치 미분방식과 비교하여 오차역전파 밥으로 구한 기울기 검증
- layer_naive.py : 곱셈계층과 덧셈계층의 구현
- train_neuralnet.py : 오차역전파법을 이용한 neralnet_train
- two_layer_net.py : 오차역전파법을 적용한 2층 신경망 클래스

[심화]

![1](https://user-images.githubusercontent.com/63804074/104217554-e46ea100-547e-11eb-8737-3ff363ef0896.jpg)
![6](https://user-images.githubusercontent.com/63804074/104217559-e59fce00-547e-11eb-8cae-729c35eb8ac4.jpg)
![7](https://user-images.githubusercontent.com/63804074/104217560-e6386480-547e-11eb-960b-a093279dea53.jpg)
![8](https://user-images.githubusercontent.com/63804074/104217563-e6386480-547e-11eb-84d3-6bf4096e4a29.jpg)
![9](https://user-images.githubusercontent.com/63804074/104217566-e6d0fb00-547e-11eb-8dc5-77aa017bbb2c.jpg)
![10](https://user-images.githubusercontent.com/63804074/104217569-e6d0fb00-547e-11eb-951d-c44c0825773d.jpg)
![11](https://user-images.githubusercontent.com/63804074/104217573-e7699180-547e-11eb-8cf1-a4c50bc788c4.jpg)
![12](https://user-images.githubusercontent.com/63804074/104217574-e8022800-547e-11eb-9c03-3c2368d9af6c.jpg)

참고사이트
[핵심 머신러닝- 뉴럴네트워크모델 2 (Backpropagation 알고리즘)
]https://www.youtube.com/watch?v=8300noBbCRU

[딥러닝 기초 - 오차역전파(back propagation) 알고리즘] https://goofcode.github.io/back-propagation
심화 내용은 핵심 머신러닝-뉴럴네트워크 모델 2 유튜브와 위 사이트를 참고했습니다.

