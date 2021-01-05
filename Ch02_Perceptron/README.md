## [2장 퍼셉트론]

퍼셉트론의 구조와 작동을 배웠습니다. AND, OR, NAND Gate가 단층 퍼셉트론 구조에서 입력 신호를 가중치에 맞추어 분류할 수 있었습니다. 하지만, XOR gate는 단층이 아닌 다층 퍼셉트론(Multi Layer Perceptron)으로 풀 수 있었습니다. 다음 장에서는 이에 대해 더 자세히 공부해 볼 것입니다.

## [정리]

- 퍼셉트론은 입출력을 갖춘 알고리즘이다. 입력을 주면 정해진 규칙에 따른 값을 출력한다.
- 퍼셉트론에서는 ‘가중치’와 ‘편향’을 매개변수로 설정한다.
- 퍼셉트론으로 AND, OR 게이트 등의 논리 회로를 표현할 수 있다.
- XOR 게이트는 단층 퍼셉트론으로는 표현할 수 없다.
- 2층 퍼셉트론을 이용하면 XOR 게이트를 표현할 수 있다.
- 단층 퍼셉트론은 직선형 영역만 표현할 수 있고, 다층 퍼셉트론은 비선형 영역도 표현할 수 있다.
- 다층 퍼셉트론은 (이론상) 컴퓨터를 표현할 수 있다.

## [파일 설명]
| File | Description |
|:--   |:--   |
|and_gate.ipynb | 단층 퍼셉트론으로 AND Gate 구현|
|nand_gate.ipynb | 단층 퍼셉트론으로 NAND Gate 구현|
|or_gate.ipynb | 단층 퍼셉트론으로 OR Gate 구현|
|xor_gate.ipynb | 다층 퍼셉트론으로 XOR Gate 구현|

## [심화]

(계단함수에서의) 다층 퍼셉트론의 수학적 의미는 무엇일까요?

⇒선형 분리 불가능한 공간에서 선형 구분 가능한 공간으로의 특징 공간 변화

<p align="center">
<img src="https://user-images.githubusercontent.com/55529617/103615414-e4960a80-4f6d-11eb-96b7-c2c8724c9586.png" width=500 height=280>
</p>

(XOR gate)를 통해 자세히 알아봅시다.

<div>
<img src="https://user-images.githubusercontent.com/55529617/103615403-e2cc4700-4f6d-11eb-995f-666c997b775b.png" width=450 height=300>
<img src="https://user-images.githubusercontent.com/55529617/103615407-e3fd7400-4f6d-11eb-8488-c834c4e9dc28.png" width=450 height=300>
</div>

여기서 활성화 함수(activation function)을 Sigmoid, Tanh와 같은 매끄러우면서 비선형 함수로 변화시킨다면 점이 아닌 영역으로 변환(특징 공간의 변형의 자유)되며 데이터를 선형분리 가능 공간으로 변환 가능합니다.

<p align="center">
<img src="https://user-images.githubusercontent.com/55529617/103615412-e4960a80-4f6d-11eb-9d34-ad9a78cfc8d5.png" width=500 height=300>
</p>

### 이미지 출처

서울시립대학교 컴퓨터과학부 "인공지능" 강의(유하진 교수님)

서울시립대학교 통계학과 "비정형데이터" 강의 (박재휘 교수님)
