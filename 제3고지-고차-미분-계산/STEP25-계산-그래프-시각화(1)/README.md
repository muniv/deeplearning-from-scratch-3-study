# STEP 25 계산 그래프 시각화(1)

계산 그래프를 시각화해보자.  
계산 그래프 시각화 장점
- 문제가 발생했을 때 원인이 되는 부분을 파악하기 쉬워진다.  
- 더 나은 계산 방법을 발견할 수도 있다.
- 신경망의 구조를 3자에게 시각적으로 전달하는 용도로도 활용할 수 있다.

시각화 도구도 밑바닥부터 만들 수 있지만, 딥러닝에서 탈선하는 느낌이라 Graphviz를 활용한다.

## 25.1 Graphviz 설치하기
https://graphviz.gitlab.io/download/


> brew install graphviz

설치가 끝나면 dot 명령어를 사용할 수 있다.
> dot -V

dot - graphviz version 8.1.0 (20230707.0739)

## 25.2 DOT 언어로 그래프 작성하기
```
digraph g{
x
y
}
```
sample.dot으로 저장

> dot sample.dot -T png -o sample.png

<image src = "../../밑바닥3 그림과 수식/그림 25-1.png">  

## 25.3 노드에 속성 지정하기
노드에 색과 모양을 지정할 수 있다.
```
digraph g{
1 [label=x, color=orange, style=filled]
2 [label=y, color=orange, style=filled]
}
```
책이랑 다르게 label에 작은 따옴표 빼야 신텍스에러 안나용
혹은 작은 따옴표 대신 큰 따옴표 사용하면 됩니다. 버전별로 다른 것 같습니다.

<image src = "../../밑바닥3 그림과 수식/그림 25-2.png">  

사각형 하늘색 노드 추가
```
digraph g{
1 [label=x, color=orange, style=filled]
2 [label=y, color=orange, style=filled]
3 [label=Exp, color=lightblue, style=filled, shape=box]
}
```
<image src = "../../밑바닥3 그림과 수식/그림 25-3.png">

이렇게 DeZero의 '변수'와 '함수' 준비 끝!

## 25.4 노드 연결하기
이제 화살표만 그리면 된다.  
화살표는 두 노드의 ID를 '->'로 연결하면 된다.
```
digraph g{
1 [label=x, color=orange, style=filled]
2 [label=y, color=orange, style=filled]
3 [label=Exp, color=lightblue, style=filled, shape=box]
1->3
3->2
}
```
<image src = "../../밑바닥3 그림과 수식/그림 25-4.png">

이제 DOT 언어로 계산 그래프 그릴 준비 끝!