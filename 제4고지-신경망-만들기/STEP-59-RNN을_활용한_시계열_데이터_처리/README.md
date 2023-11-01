# RNN을 활용한 시계열 데이터 처리
피드 포워드(feed forward)구조의 신경망: 신호가 한 방향으로만 흘러가기 때문에 입력 신호만으로 출력을 결정한다는 특징이 있다.  
한편, 순환 신경망(Recurrent Neural Network)은 순환 구조를 가지고 있다.

<image src = "../../밑바닥3 그림과 수식/그림 59-1.png">

RNN을 DeZero를 이용하여 구현해보자

## 59.1 RNN 계층 구현
아래는 RNN 순전파를 표시한 수식이다.

<image src = "../../밑바닥3 그림과 수식/식 59.1.png">

```python
class RNN(Layer):
    def __init__(self, hidden_size, in_size=None):
        """An Elman RNN with tanh.

        Args:
            hidden_size (int): The number of features in the hidden state.
            in_size (int): The number of features in the input. If unspecified
            or `None`, parameter initialization will be deferred until the
            first `__call__(x)` at which time the size will be determined.

        """
        super().__init__()
        self.x2h = Linear(hidden_size, in_size=in_size)
        self.h2h = Linear(hidden_size, in_size=in_size, nobias=True)
        self.h = None

    def reset_state(self):
        self.h = None

    def forward(self, x):
        if self.h is None:
            h_new = F.tanh(self.x2h(x))
        else:
            h_new = F.tanh(self.x2h(x) + self.h2h(self.h))
        self.h = h_new
        return h_new

```

- x2h: 입력 x에서 은닉 상태 h로 변환하는 완전연결계층
- h2h: 이전 은닉 상태에서 다음 은닉 상태로 변환하는 완전연결계층
- forward 메서드는 self.h(은닉 상태)유무에 따라 처리 방식 달라짐
```python
import numpy as np
import dezero.layers.as L

rnn = L.RNN(10) # 은닉층 크기 지정
x = np.random.rand(1,1)
h = rnn(x)
print(h.shape)
```
```text
(1, 10)
```
<image src = "../../밑바닥3 그림과 수식/그림 59-2.png">
<image src = "../../밑바닥3 그림과 수식/그림 59-3.png">


## 59.2 RNN 모델 구현
```python
from dezero import Model
import dezero.functions as F
import dezero.layers as L

class SimpleRNN(Model):
    def __init__(self, hidden_size, out_size):
        super().__init__()
        self.rnn = L.RNN(hidden_size)
        self.fc = L.Linear(out_size)

    def reset_state(self):
        self.rnn.reset_state()

    def __call__(self, x):
        h = self.rnn(x)
        y = self.fc(h)
        return y

```
- fc에 linear계층 추가
- reset_state 메서드: RNN계층의 은닉 상태를 재설정

모델을 학습해보자
```python
seq_data = [np.random.randn(1, 1) for _ in range(1000)] 
xs = seq_data[0:-1]
ts = seq_data[1:]

model = SimpleRNN(10, 1)

loss, cnt = 0, 0
for x, t in zip(xs, ts):
    y = model(x)
    loss += F.mean_squared_error(y, t)

    cnt += 1
    # 두 번째 데이터가 들어왔을 때 역전파 수행
    if cnt == 2:
        model.cleargrads()
        loss.backward()
        break

```
<image src = "../../밑바닥3 그림과 수식/그림 59-4.png">

계산 그래프 만들어진 후, loss.backward()를 통해 각 파라미터의 그라디언트 구할 수 있다.  
이렇게 일련의 입력 데이터로 구성된 계산 그래프에서의 역전파를 BPTT(Backpropagation Through Time)이라고 한다.  

## 59.3 '연결'을 끊어주는 메서드
역전파를 잘하려면 계산 그래프를 적당한 길이에서 끊어줘야 한다. 
긴 시계열 데이터를 학습할 때 문제: 시계열 데이터의 시간 크기가 커지는 것에 비례하여 BPTT가 소비하는 컴퓨팅 자원도 증가. 시간 크기가 커지면 역전파 시 기울기가 불안정해지기도 한다.  
 
이것을 Truncated BPTT라고 한다.  
Truncated BPTT를 수행할 때에는 RNN의 은닉 상태가 유지되어야 한다는 점에 주의해야 한다.  
따라서 Variable 클래스에 연결을 끊어주는 메서드가 필요해진다.  
```python
class Variables:
    ...
     def unchain(self):
        self.creator = None
    ...
    def unchain_backward(self):
        if self.creator is not None:
            funcs = [self.creator]
            while funcs:
                f = funcs.pop()
                for x in f.inputs:
                    if x.creator is not None:
                        funcs.append(x.creator)
                        x.unchain()
```
unchain_backward 메서드는 호출된 변수에서 시작하여 계산 그래프를 거슬러 올라가며 마주치는 모든 변수의 unchain 메서드를 호출한다.  


## 59.4 사인파 예측
```python
# Hyperparameters
max_epoch = 100
hidden_size = 100
bptt_length = 30

train_set = dezero.datasets.SinCurve(train=True)
seqlen = len(train_set)

model = SimpleRNN(hidden_size, 1)
optimizer = dezero.optimizers.Adam().setup(model)

# Start training.
for epoch in range(max_epoch):
    model.reset_state()
    loss, count = 0, 0

    for x, t in train_set:
        x = x.reshape(1, 1) # 형상을 (1, 1)로 변환
        y = model(x)
        loss += F.mean_squared_error(y, t)
        count += 1

        # Truncated BPTT의 타이밍 조정 - backward 메서드를 언제 호출할지 결정
        if count % bptt_length == 0 or count == seqlen:
            model.cleargrads()
            loss.backward()
            loss.unchain_backward() # 연결 끊기
            optimizer.update()

    avg_loss = float(loss.data) / count
    print('| epoch %d | loss %f' % (epoch + 1, avg_loss))
```
```python
# Plot
xs = np.cos(np.linspace(0, 4 * np.pi, 1000))
model.reset_state()
pred_list = []

with dezero.no_grad():
    for x in xs:
        x = np.array(x).reshape(1, 1)
        y = model(x)
        pred_list.append(float(y.data))

plt.plot(np.arange(len(xs)), xs, label='y=cos(x)')
plt.plot(np.arange(len(xs)), pred_list, label='predict')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.show()
```

<image src = "../../밑바닥3 그림과 수식/그림 59-7.png">