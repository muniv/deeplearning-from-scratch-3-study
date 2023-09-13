# STEP 35 고차 미분 계산 그래프

이번에 추가할 함수는 tanh이다.  
<image src = "../../밑바닥3 그림과 수식/식 35.1.png">

<image src = "../../밑바닥3 그림과 수식/그림 35-1.png">  
tanh는 입력을 -1 ~ 1 사이의 값으로 변환한다.

## 35.1 tanh 함수 미분
tanh 함수의 미분은 분수 함수의 미분 공식을 사용해서 계산한다.
<image src = "../../밑바닥3 그림과 수식/식 35.2.png">

<image src = "../../밑바닥3 그림과 수식/식 35.3.png">

## 35.2 tanh 함수 구현
```python
class Tanh(Function):
    def forward(self, x):
        xp = cuda.get_array_module(x)
        y = xp.tanh(x)
        return y

    def backward(self, gy):
        y = self.outputs[0]()  # weakref
        gx = gy * (1 - y * y)
        return gx


def tanh(x):
    return Tanh()(x)
```

## 35.3 고차 미분 계산 그래프 시각화

```python
import numpy as np
from dezero import Variable
from dezero.utils import plot_dot_graph
import dezero.functions as F

x = Variable(np.array(1.0))
y = F.tanh(x)
x.name = 'x'
y.name = 'y'
y.backward(create_graph=True)

iters = 1

for i in range(iters):
    gx = x.grad
    x.cleargrad()
    gx.backward(create_graph=True)

gx = x.grad
gx.name = 'gx' + str(iters + 1)
plot_dot_graph(gx, verbose=False, to_file='tanh.png')
```

iters=0 즉, 1차 미분 계산 그래프는 아래와 같다.  
<image src = "../../밑바닥3 그림과 수식/그림 35-2.png">  
DeZero의 Tanh, Mul, Sub가 사용되고 있는 것을 확인할 수 있다.  

이어서 iters를 늘려서 2차~5차 미분을 계산해보자.  
<image src = "../../밑바닥3 그림과 수식/그림 35-3.png">   
차수가 늘어날수록 계산 그래프도 복잡해진다.  
역전파를 할 때마다 기존까지의 계산에 대한 새로운 계산 그래프가 만들어지므로 노드 수가 기하급수적으로 증가하게 된다.  

6차 미분, 7차 미분의 결과는 아래와 같다.
<image src = "../../밑바닥3 그림과 수식/그림 35-4.png">  

8차 미분의 결과  
<image src = "../../밑바닥3 그림과 수식/그림 35-5.png">  

[고화질 이미지](https://github.com/WegraLee/deep-learning-from-scratch-3/tree/tanh)
