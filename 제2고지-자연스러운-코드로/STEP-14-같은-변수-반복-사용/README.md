# STEP 14 같은 변수 반복 사용

<image src = "../../밑바닥3 그림과 수식/그림 14-1.png">  

y = add(x, x)처럼 같은 변수를 반복해서 사용하는 경우 제대로 작동하게 될까?

```python
x = Variable(np.array(3.0))
y = add(x, x)
print('y', y.data)

y.backward()
print('x.grad', x.grad)
```
실행 결과
```text
y 6.0
x.grad 1.0
```
y의 값은 잘 나왔는데, x.grad는 잘못 나왔다.(y=2x이므로 미분값은 2가 되어야 함)

## 14.1 문제의 원인
```python
class Variable:
    ...

    def backward(self):
        if self.grad is None:
            self.grad = np.ones_like(self.data)

        funcs = [self.creator]
        while funcs:
            f = funcs.pop()
            gys = [output.grad for output in f.outputs]
            gxs = f.backward(*gys)
            if not isinstance(gxs, tuple):
                gxs = (gxs,)

            for x, gx in zip(f.inputs, gxs):
                x.grad = gx # 여기가 실수!

                if x.creator is not None:
                    funcs.append(x.creator)
```

현재 코드에서는 출력쪽에서 전해지는 미분값을 그대로 대입한다.   
따라서 같은 변수를 반복해서 사용하면 전파되는 미분값이 덮어써진다!

<image src = "../../밑바닥3 그림과 수식/그림 14-2.png">

이때 미분은 1+1 = 2 가 되어야 하는데 지금은 그냥 덮어쓰고 있으므로 1이 나온 것이다.

## 14.2 해결책
요렇게 수정하면 된다!
```python
class Variable:
    ...

    def backward(self):
        if self.grad is None:
            self.grad = np.ones_like(self.data)

        funcs = [self.creator]
        while funcs:
            f = funcs.pop()
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
                    funcs.append(x.creator)
```
