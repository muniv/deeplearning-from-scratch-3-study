# STEP 15 복잡한 계산 그래프(이론 편)


<image src = "../../밑바닥3 그림과 수식/그림 15-1.png">  

변수와 함수가 꼭 이렇게 한 줄로 연결되리라는 법은 없다.  
아래와 같은 계산 그래프도 나올 수 있다!

<image src = "../../밑바닥3 그림과 수식/그림 15-2.png">  

아직 우리의 DeZero는 이런 복잡한 연결을 처리하지 못한다.(미분을 계산하지 못하고, 역전파를 제대로 못한다)

## 15.1 역전파의 올바른 순서
DeZero가 왜 처리를 못할까? 원인을 살펴보자.  
아래 처럼 간단해보이지만 중간에 분기했다가 합류하는 계산 그래프도 지금 DeZero로는 제대로 미분하지 못한다.
<image src = "../../밑바닥3 그림과 수식/그림 15-3.png">  

위 그림에서 주목할 부분은 변수 a이다. 역전파 때는 출력 쪽에서 전파되는 미분값을 더해야 하므로 a의 미분을 계산하려면 a의 출력 쪽, 즉 B와 C에서 전파되는 미분 값이 필요하다.   
역전파의 순서를 따지자면 D -> B , C -> A 의 순서로 진행된다.
<image src = "../../밑바닥3 그림과 수식/그림 15-4.png">  

## 15.2 현재의 DeZero
현재 우리 DeZero는 위와 같은 순서로 역전파하고 있나 확인해보자.
```python
class Variable:
    ...

    def backward(self):
        if self.grad is None:
            self.grad = np.ones_like(self.data)

        funcs = [self.creator] # 가장 먼저 y의 creator가 추가된다.
        while funcs:
            f = funcs.pop() # 다음에 처리할 함수를 리스트의 끝에서 꺼낸다
            gys = [output.grad for output in f.outputs]
            gxs = f.backward(*gys)
            if not isinstance(gxs, tuple):
                gxs = (gxs,)

            for x, gx in zip(f.inputs, gxs):
                if x.grad is None:
                    x.grad = gx
                else:
                    x.grad = x.grad + gx

                if x.creator is not None:
                    funcs.append(x.creator) # 처리할 함수의 후보를 funcs리스트의 끝에 추가한다. 

```

### 첫 번째 반복
코드에 따르면 while문 직전에 y의 creator인 D가 funcs에 추가된 채로 시작한다.  
while문 끝에서 에서 funcs 리스트는 [B, C]가 된다.
<image src = "../../밑바닥3 그림과 수식/그림 15-6.png">  
  
### 두 번째 반복
다음 while문 사이클에서 funcs.pop()을 하면 마지막 원소인 C가 꺼내진다.  
그리고 while문 끝에서 C의 input인 c의 creator인 A가 리스트에 추가된다.   
이때 funcs 리스트는 [B, A]이다. 
<image src = "../../밑바닥3 그림과 수식/그림 15-7.png">  

### 세 번째 반복
이제 다음 while문 사이클에서 funcs.pop()을 하면 마지막 원소인 A가 꺼내지고  
while문 끝에서 A의 input인 x의 creator가 없으니 추가되는 함수 원소가 없이 다음 턴으로 넘어가게 된다.  
이때 funcs 리스트는 [B]이다.

### 네 번째 반복
다음 while문 사이클에서 funcs.pop()을 하면 B가 꺼내지고,  
while문 끝에서 B의 input인 a의 creator인 A가 funcs에 append 되어서  
이때 funcs 리스트는 [A]가 된다.

### 다섯 번째 반복
마지막으로 다음 while문 사이클에서 funcs.pop()을 하면 A가 꺼내지고,
while문 끝에서 A의 input인 x의 creator가 없으니 추가되는 함수 원소가 없고,  
funcs 리스트는 빈 리스트가 되어 while문이 종료된다.  

위 과정에서 함수 처리 순서를 정리해보면 D -> C -> A -> B -> A 가 된다.  
<image src = "../../밑바닥3 그림과 수식/그림 15-5.png">  

문제는 세 번째 반복에서 일어났다. B를 꺼내야 했는데 A를 꺼낸 것이다.  
지금까지 우리는 한 줄로 나열된 계산 그래프를 다뤘기 때문에 함수의 순서를 고려하지 않아도 되었다. 하지만 이제는 순서를 고려해야 한다.  

## 15.3 함수 우선순위
funcs 리스트에는 다음에 처리할 함수의 '후보'들이 들어가게 된다.  
지금까지는 역전파를 처리할 때 아무생각없이 그 후보 중 마지막 원소만 꺼냈다.  
그러나 이제는 후보 함수 중 어떤 것을 먼저 꺼내야 할지 우선순위를 주어야 한다.  
아까 세번째 반복에서 A대신 B를 꺼낼 수 있게 말이다!  

그렇다면 어떻게 우선순위를 주어야 할까?  
순전파 때 우리는 '부모-자식 관계'를 파악할 수 있었는데, 이를 세대로 표현할 수 있다.
<image src = "../../밑바닥3 그림과 수식/그림 15-8.png">  

이 세대를 우선순위로 활용하면 된다.  
역전파 시 세대가 큰 쪽부터 처리하면 '부모'보다 '자식'이 먼저 처리됨을 보장할 수 있다.  
위 그림에서 함수A는 0세대, 함수B는 1세대에 속한다.  
아까 문제가 되었던 세 번째 반복에서 funcs = [B, A] 일 때 마지막 원소를 꺼내는 것이 아닌, 세대가 큰 B부터 꺼내면 이제 역전파 함수 처리 순서가 제대로 되게 된다.
