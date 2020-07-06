##### 公式4.1-4.2

$$
\begin{aligned}
z &=\sum_{i=1}^{d} w_{i} x_{i}+b \\
&=\boldsymbol{w}^{\mathrm{T}} \boldsymbol{x}+b
\end{aligned}
$$

**实例4.1-4.2**

```python
import numpy as np

z = 0  # 净输入
w = np.asarray([1, 2, 3])  # 3维权重向量
x = np.asarray([1, 2, 3])  # 输入向量
b = 1  # 偏置

z = np.vdot(w, x) + b  # 公式4.1，4.2

print(z)

'''
15
'''
```

##### 公式4.3

$$
a=f(z)
$$
**实例4.3**

```python
import numpy as np


def f(x):  # 非线性函数f，假设为正弦函数
    return np.sin(x)


a = 0  # 活性值
z = 15  # 净输入

a = f(15)  # 公式4.3

print(a)

'''
0.6502878401571168
'''
```

##### 公式4.4

$$
\sigma(x)=\frac{1}{1+\exp (-x)}
$$

**实例4.4**

```python
import numpy as np
import math


def sigma(x):  # Logistic函数
    return 1 / (1 + math.exp(-x))


sigma_x = 0  # 输出
x = 1  # 输入

sigma_x = sigma(x)  # 公式4.4

print(sigma_x)

'''
0.7310585786300049
'''
```

##### 公式4.5

$$
\tanh (x)=\frac{\exp (x)-\exp (-x)}{\exp (x)+\exp (-x)}
$$

**实例4.5**

```python
import numpy as np
import math


def tanh(x):  # Tanh函数
    return (math.exp(x) - math.exp(-x)) / (math.exp(x) + math.exp(-x))


tanh_x = 0  # 输出
x = 1  # 输入

tanh_x = tanh(x)  # 公式4.5

print(tanh_x)

'''
0.7615941559557649
'''
```

##### 公式4.6

$$
\tanh (x)=2 \sigma(2 x)-1
$$

**实例4.6**

```python
import numpy as np
import math


def sigma(x):  # Logistic函数
    return 1 / (1 + math.exp(-x))


def tanh(x):  # Tanh函数
    return 2 * sigma(2 * x) - 1


tanh_x = 0  # 输出
x = 1  # 输入

tanh_x = tanh(x)  # 公式4.6

print(tanh_x)

'''
0.7615941559557646
'''
```

##### 公式4.7-4.8

$$
\begin{aligned}
g_{l}(x) & \approx \sigma(0)+x \times \sigma^{\prime}(0) \\
&=0.25 x+0.5
\end{aligned}
$$

**实例4.7-4.9**

```python
import numpy as np
import math


def sigma(x):  # Logistic函数
    return 1 / (1 + math.exp(-x))


def sigma_fd(x):  # Logistic函数的一阶导函数
    return sigma(x)*(1-sigma(x))


def g_l(x):  # Logistic函数的一阶泰勒展开函数
    return sigma(0) + x * sigma_fd(0)


g_l_x = 0  # 输出
x = 1  # 输入

g_l_x = g_l(x)  # 公式4.7-4.8

print(g_l_x)

'''
0.75
'''
```

##### 公式4.9-4.11

$$
\begin{aligned}
\text { hard-logistic }(x) &=\left\{\begin{array}{ll}
1 & g_{l}(x) \geq 1 \\
g_{l} & 0<g_{l}(x)<1 \\
0 & g_{l}(x) \leq 0
\end{array}\right.\\
&=\max \left(\min \left(g_{l}(x), 1\right), 0\right) \\
&=\max (\min (0.25 x+0.5,1), 0)
\end{aligned}
$$

**实例4.9-4.11**

```python
import numpy as np
import math


def sigma(x):  # Logistic函数
    return 1 / (1 + math.exp(-x))


def sigma_fd(x):  # Logistic函数的一阶导函数
    return sigma(x)*(1-sigma(x))


def g_l(x):  # Logistic函数的一阶泰勒展开函数
    return sigma(0) + x * sigma_fd(0)


def hard_logistic(x):  # 分段函数hard_logistic近似logistic函数
    if g_l(x) >= 1:
        return 1
    elif g_l(x) <= 0:
        return 0
    else:
        return g_l(x)


hard_logistic_x = 0  # 输出
x = 1  # 输入

hard_logistic_x = hard_logistic(x)  # 公式4.9-4.11

print(hard_logistic_x)

'''
0.75
'''
```

##### 公式4.12-4.13

$$
\begin{aligned}
g_{t}(x) & \approx \tanh (0)+x \times \tanh ^{\prime}(0) \\
&=x
\end{aligned}
$$

**实例4.12-4.13**

```python
import numpy as np
import math


def tanh(x):  # Tanh函数
    return (math.exp(x) - math.exp(-x)) / (math.exp(x) + math.exp(-x))


def tanh_fd(x):  # Tanh函数的一阶导函数
    return 1 - tanh(x) ** 2


def g_t(x):  # Tanh函数在0附近的一阶泰勒展开函数
    return tanh(0) + x * tanh_fd(0)


g_t_x = 0  # 输出
x = 1  # 输入

g_t_x = g_t(x)  # 公式4.12-4.13

print(g_t_x)

'''
1.0
'''
```

##### 公式4.14-4.15

$$
\begin{aligned}
\text { hard-tanh }(x) 
&=\left\{\begin{array}{ll}
1 & g_{t}(x) \geq 1 \\
g_{t} & 0<g_{t}(x)<1 \\
-1 & g_{t}(x) \leq -1
\end{array}\right.\\
&=\max \left(\min \left(g_{t}(x), 1\right),-1\right) \\
&=\max (\min (x, 1),-1)
\end{aligned}
$$

**实例4.14-4.15**

```python
import numpy as np
import math


def tanh(x):  # Tanh函数
    return (math.exp(x) - math.exp(-x)) / (math.exp(x) + math.exp(-x))


def tanh_fd(x):  # Tanh函数的一阶导函数
    return 1 - tanh(x) ** 2


def g_t(x):  # Tanh函数在0附近的一阶泰勒展开函数
    return tanh(0) + x * tanh_fd(0)


def hard_tanh(x):  # 分段函数hard_tanh近似tanh函数
    if g_t(x) >= 1:
        return 1
    elif g_t(x) <= -1:
        return - 1
    else:
        return g_t(x)


hard_tanh_x = 0  # 输出
x = 1  # 输入

hard_tanh_x = hard_tanh(x)  # 公式4.14-4.15

print(hard_tanh_x)

'''
1
'''
```

##### 公式4.16-4.17

$$
\begin{aligned}
\operatorname{ReLU}(x) &=\left\{\begin{array}{ll}
x & x \geq 0 \\
0 & x<0
\end{array}\right.\\
&=\max (0, x)
\end{aligned}
$$

**实例4.16-4.17**

```python
import numpy as np
import math


def relu(x):  # ReLU函数,修正线性单元函数
    if x >= 0:
        return x
    else:
        return 0


relu_x = 0  # 输出
x = 1  # 输入

relu_x = relu(x)  # 公式4.16-4.17

print(relu_x)

'''
1
'''
```

##### 公式4.18-4.19

$$
\begin{aligned}
\text { LeakyReLU }(x) &=\left\{\begin{array}{ll}
x & \text { if } x>0 \\
\gamma x & \text { if } x \leq 0
\end{array}\right.\\
&=\max (0, x)+\gamma \min (0, x)
\end{aligned}
$$

**实例4.18-4.19**

```python
import numpy as np
import math


def leaky_relu(x):  # 带泄露的Relu函数
    gamma = 0.001
    if x > 0:
        return x
    else:
        return gamma * x


leaky_relu_x = 0  # 输出
x = 1  # 输入

leaky_relu_x = leaky_relu(x)  # 公式4.18-4.19

print(leaky_relu_x)

'''
1
'''
```

##### 公式4.20

$$
\text { LeakyReLU }(x)=\max (x, \gamma x)
$$

**实例4.20**

```python
import numpy as np
import math


def leaky_relu(x):  # 带泄露的Relu函数
    gamma = 0.001
    return max(x, gamma * x)

leaky_relu_x = 0  # 输出
x = 1  # 输入

leaky_relu_x = leaky_relu(x)  # 公式4.20

print(leaky_relu_x)

'''
1
'''
```

##### 公式4.21-4.22

$$
\begin{aligned}
\operatorname{PReLU}_{i}(x) &=\left\{\begin{array}{ll}
x & \text { if } x>0 \\
\gamma_{i} x & \text { if } x \leq 0
\end{array}\right.\\
&=\max (0, x)+\gamma_{i} \min (0, x)
\end{aligned}
$$

**实例4.21-4.22**

```python
import numpy as np
import math

gamma = [0.001, 0.002, 0.003]


def p_relu(i, x):  # 带参数的Relu函数
    if x > 0:
        return x
    else:
        return gamma[i] * x


p_relu_i_x = 0  # 输出
x = 1  # 输入
i = 1  # 神经元的序号


p_relu_i_x = p_relu(i, x)  # 公式4.21-4.22

print(p_relu_i_x)

'''
1
'''
```

##### 公式4.23-4.24

$$
\begin{aligned}
\operatorname{ELU}(x) &=\left\{\begin{array}{ll}
x & \text { if } x>0 \\
\gamma(\exp (x)-1) & \text { if } x \leq 0
\end{array}\right.\\
&=\max (0, x)+\min (0, \gamma(\exp (x)-1))
\end{aligned}
$$

**实例4.23-4.24**

```python
import numpy as np
import math


def elu(x):  # ELU函数，指数线性单元函数
    gamma = 0.001
    if x > 0:
        return x
    else:
        return gamma*(math.exp(x)-1)


elu_x = 0  # 输出
x = 1  # 输入

elu_x = elu(x)  # 公式4.23-4.24

print(elu_x)

'''
1
'''
```

##### 公式4.25

$$
\text { Softplus }(x)=\log (1+\exp (x))
$$

**实例4.25**

```python
import numpy as np
import math


def softplus(x):  # Softplus函数
    return math.log(1 + math.exp(x))


softplus_x = 0  # 输出
x = 1  # 输入

softplus_x = softplus(x)  # 公式4.25

print(softplus_x)

'''
1.3132616875182228
'''
```

##### 公式4.26

$$
\operatorname{swish}(x)=x \sigma(\beta x)
$$

**实例4.26**

```python
import numpy as np
import math


def sigma(x):  # Logistic函数
    return 1 / (1 + math.exp(-x))


def swish(x):  # swish函数
    # beta = 0  #线性函数
    beta = 1  # x>0线性，x<0近似饱和
    # beta = 100  #近似relu函数
    return x * sigma(beta * x)


swish_x = 0  # 输出
x = 1  # 输入

swish_x = swish(x)  # 公式4.26

print(swish_x)
'''
0.7310585786300049
'''
```

##### 公式4.27

$$
\operatorname{GELU}(x)=x P(X \leq x)
$$

**实例4.27**

```python
import numpy as np
import math
from scipy.stats import norm


def P(x):  # 高斯分布N(mu,sigma^2)的累积分布函数
    return norm.cdf(x, 0, 1)


def gelu(x):  # GELU函数，高斯误差线性单元
    return x * P(x)


gelu_x = 0  # 输出
x = 1  # 输入

gelu_x = gelu(x)  # 公式4.27

print(gelu_x)

'''
0.8413447460685429
'''
```

##### 公式4.28

$$
\operatorname{GELU}(x) \approx 0.5 x\left(1+\tanh \left(\sqrt{\frac{2}{\pi}}\left(x+0.044715 x^{3}\right)\right)\right)
$$

**实例4.28**

```python
import numpy as np
import math


def tanh(x):  # Tanh函数
    return (math.exp(x) - math.exp(-x)) / (math.exp(x) + math.exp(-x))


def gelu(x):  # 近似GELU函数
    return 0.5 * x * (1 + tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * x ** 3)))


gelu_x = 0  # 输出
x = 1  # 输入

gelu_x = gelu(x)  # 公式4.28

print(gelu_x)
'''
0.8411919906082768
'''
```

##### 公式4.29

$$
\operatorname{GELU}(x) \approx x \sigma(1.702 x)
$$

**实例4.29**

```python
import numpy as np
import math


def sigma(x):  # Logistic函数
    return 1 / (1 + math.exp(-x))


def gelu(x):  # 近似GELU函数
    return x * sigma(1.702 * x)


gelu_x = 0
x = 1

gelu_x = gelu(x)

print(gelu_x)
'''
0.8457957659328212
'''
```

##### 公式4.30

$$
z_{k}=\boldsymbol{w}_{k}^{\mathrm{T}} \boldsymbol{x}+b_{k}
$$

**实例4.30**

```python
import numpy as np
import math

K = 3  # 权重向量的个数
z = np.zeros(K)  # K个净输入组成的向量
w = np.asarray([[1, 2, 3], [1, 2, 3], [1, 2, 3]])  # K个权重向量组成的矩阵
x = np.asarray([1, 2, 3])  # 输入向量
b = np.asarray([1, 2, 3])  # K个偏置组成的向量

for k in range(K):
    z[k] = np.vdot(w[k], x) + b[k]  # 公式4.30

print(z)

'''
[15. 16. 17.]
'''
```

##### 公式4.31

$$
\operatorname{maxout}(\boldsymbol{x})=\max _{k \in[1, K]}\left(z_{k}\right)
$$

**实例4.31**

```python
import numpy as np
import math

K = 3  # 权重向量的个数
z = np.zeros(K)  # K个净输入组成的向量
w = np.asarray([[1, 2, 3], [1, 2, 3], [1, 2, 3]])  # K个权重向量组成的矩阵
x = np.asarray([1, 2, 3])  # 输入向量
b = np.asarray([1, 2, 3])  # K个偏置组成的向量

for k in range(K):
    z[k] = np.vdot(w[k], x) + b[k]  # 公式4.30


def maxout(x):  # maxout函数
    return max(x)


maxout_x = 0  # 输出
x = 1  # 输入

maxout_x = maxout(z)  # 公式4.31

print(maxout_x)
'''
17.0
'''
```



