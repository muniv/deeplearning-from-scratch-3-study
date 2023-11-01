# STEP 51 MNIST 학습

Dataset 클래스로 데이터셋 처리를 위한 공통 인터페이스를 마련했고, '전처리'를 설정할 수 있도록 했다.  
DataLoader 클래스로는 Dataset에서 미니배치 단위로 데이터를 꺼내올 수 있게 했다.   

<image src = "../../밑바닥3 그림과 수식/그림 51-1.png"> 

## 51.1 MNIST 데이터셋
dezero/datasets.py
```python
import dezero

train_set = dezero.datasets.MNIST(train=True, transform=None)
train_set = dezero.datasets.MNIST(train=False, transform=None)

print(len(train_set))
print(len(test_set))
```
```
60000
10000
```
---

```python
x, t = train_set[0]
print(type(x), x.shape)
print(t)
```
```
<class 'numpy.ndarray> (1, 28, 28)
5
```
---
```python
import matplotlib.pyplot as plt

x, t = train_set[0]
plt.imshow(x.reshape(28,28),cmap='gray')
plt.axis('off')
plt.show()
```
<image src = "../../밑바닥3 그림과 수식/그림 51-2.png">

수행할 전처리
```python
def(x):
    x = x.flatten() # (784,)로 평탄화
    x = x.astype(np.float32) # 32비트 부동소수점으로 변환
    x /= 255.0 # 범위가 0.0~1.0사이가 되도록
    return x

train_set = dezero.datasets.MNIST(train=True, transform=f)
train_set = dezero.datasets.MNIST(train=False, transform=f)
```
MNIST클래스에서는 위 전처리가 디폴트여서 dezero.datasets.MNIST(train=True)로 호출하면 전처리가 자동으로 수행된다.

## 51.2 MNIST 학습하기

```python
####추가
max_epoch = 5
batch_size = 100
hidden_size = 1000

train_set = dezero.datasets.MNIST(train=True)
test_set = dezero.datasets.MNIST(train=False)
####추가
train_loader = DataLoader(train_set, batch_size)
test_loader = DataLoader(test_set, batch_size, shuffle=False)

model = MLP((hidden_size, 10))
optimizer = optimizers.SGD().setup(model)
#model = MLP((hidden_size, hidden_size, 10), activation=F.relu)
#optimizer = optimizers.Adam().setup(model)

for epoch in range(max_epoch):
    sum_loss, sum_acc = 0, 0

    for x, t in train_loader:
        y = model(x)
        loss = F.softmax_cross_entropy(y, t)
        acc = F.accuracy(y, t)
        model.cleargrads()
        loss.backward()
        optimizer.update()

        sum_loss += float(loss.data) * len(t)
        sum_acc += float(acc.data) * len(t)

    print('epoch: {}'.format(epoch+1))
    print('train loss: {:.4f}, accuracy: {:.4f}'.format(
        sum_loss / len(train_set), sum_acc / len(train_set)))

    sum_loss, sum_acc = 0, 0
    with dezero.no_grad():
        for x, t in test_loader:
            y = model(x)
            loss = F.softmax_cross_entropy(y, t)
            acc = F.accuracy(y, t)
            sum_loss += float(loss.data) * len(t)
            sum_acc += float(acc.data) * len(t)

    print('test loss: {:.4f}, accuracy: {:.4f}'.format(
        sum_loss / len(test_set), sum_acc / len(test_set)))
```
결과: 테스트 데이터셋에서 약 86%

## 51.3 모델 개선하기
활성화 함수: 시그모이드 -> ReLU
최적화 기법: SGD -> Adam

```python
class ReLU(Function):
    def forward(self, x):
        xp = cuda.get_array_module(x)
        y = xp.maximum(x, 0.0)
        return y

    def backward(self, gy):
        x, = self.inputs
        mask = x.data > 0
        gx = gy * mask
        return gx


def relu(x):
    return ReLU()(x)
```

결과: 학습 데이터셋에서 약 99%, 테스트 데이터셋에서 약 98%

---
!4고지 끝!  
지금까지 배운 지식이 파이토치, 체이너 같은 프레임워크에도 통한다.  
[파이토치 MNIST 코드](https://github.com/pytorch/examples/blob/main/mnist/main.py)

---
## 칼럼: 딥러닝 프레임워크
파이토치, 체이너, 텐서플로우 등 인기 프레임워크들 설계 사상의 공통점
- Define-by-Run 방식으로 계산 그래프를 만들 수 있다.
- 사전 정의된 다양한 함수와 계층을 제공한다.
- 다양한 매개변수 갱신 클래스(옵티마이저)를 제공한다.
- 모델은 상속을 통해 구현할 수 있다.
- 데이터셋을 관리하는 클래스를 제공한다.
- CPU 외에도 GPU나 특정한 AISC(주문형 반도체)도 활용할 수 있다.
- 성능 향상을 위해(실제 제품 적용 대비) Define-and-Run 모드 실행도 지원

### *프레임워크와 라이브러리의 차이
라이브러리: 편의 함수와 데이터 구조의 모음, 사용자는 라이브러리에서 필요한 것을 적당히 꺼내 사용. 이 과정에서 프로그램 제어는 사용자가 정한다.  
프레임워크: 전체의 토대를 제공. 딥러닝 프레임워크라면 자동 미분의 기초를 제공하고 사용자는 그 위에 필요한 계산을 구축한다. 전체적인 역할은 프레임워크가 담당. 
<image src = "IMG_8880.png"> 

### Define-by-Run 방식의 자동 미분
딥러닝 프레임워크에서 가장 중요한 기능은 '자동 미분' 
자동 미분 덕분에 귀찮은 계산과 코드작성 없이 즉시 미분값 계산 가능  
현대 프레임워크는 계산 그래프를 Define-by-Run 방식으로 만들어주는데, 코드는 즉시 실행되고 동시에 뒤편에서는 계산 그래프가 만들어 지는 것이다.  
예) 파이토치
```pytorch
import torch

x = torch.randn(1, requires_grad=True)

y = x

for i in range(5):
    y = y*2

y.backward()
print(x.grad)
```
```
tensor([32.])
```
파이토치에서 텐서를 표현하는 클래스는 Tensor(DeZero의 Variable에 해당함)

파이토치, 체이너, 텐서플로우는 Define-by-Run말고 Define-and-Run(정적 계산 그래프)모드도 지원한다. (텐서플로우는 2.x버전부터 가능)   
Define-and-Run은 성능이 요구되는 실제 제품(서비스)이나 엣지 환경에서 사용할 때 적합하다.  
(왜? 먼저 그래프를 정의하고 나중에 실행함으로써 최적화 가능?)

### 계층 컬렉션
<image src = "../../밑바닥3 그림과 수식/그림 D-1.png">
여기서 중요한 점은 이러한 계층 컬렉션이 '자동 미분' 구조 위에 구축된다는 사실.
<image src = "../../밑바닥3 그림과 수식/그림 D-2.png">
위가 딥러닝 프레임워크 기본 구조. 

### 옵티마이저 컬렉션
<image src = "../../밑바닥3 그림과 수식/그림 D-3.png">
다양한 옵티마이저를 제공함으로써 우리는 파라미터 업데이트 방식을 다양하게 생각할 수 있고, 다른 옵티마이저로 전환하기도 쉬워서 실험 검증 반복이 편해진다. 

### 정리
자동 미분 기능을 기반으로 그 위에 다양한 계층 컬렉션이 존재한다. 그리고 파라미터를 업데이트하는 옵티마이저 컬렉션도 있다. 
<image src = "../../밑바닥3 그림과 수식/그림 D-4.png">
위 세 가지 요소는 거의 모든 프레임워크가 제공하는 중요한 기능.

<image src = "DeZeroClasses.png">