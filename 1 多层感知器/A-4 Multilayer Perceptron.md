#### 感知器范例（一）：AND逻辑元

* 逻辑图：

​                     **x1**--------------->      

​										        **|AND逻辑单元)** ------------>out

​                     **x2**--------------->         

* 真值表：

  |  x1  |  x2  | out  |
  | :--: | :--: | :--: |
  |  0   |  0   |  0   |
  |  0   |  1   |  0   |
  |  1   |  0   |  0   |
  |  1   |  1   |  1   |

* 感知器图例：

  ​                 **x1**        ------  **w1=1**  ---->

  ​                 **x2**        ------  **w2=1 ** ---->        **AND感知器**     -------->out

  ​			                o----  **b=-0.5 ** ---->                ↓

  ​															  	Σwx + b > 0    ->  1

​                                                                         Σwx + b <= 0  ->  0

* 演算过程

  * 第一列：0，0，0：

    0x1 + 0x1 +（-0.5）= -0.5  <= 0，输出0，和真值表相同，故不用调整权值和偏向量

  * 第二列：0，1，0：

    0x1 + 1x1 +（-0.5）= 0.5  > 0，输出1，和真值表不同，需要调整权重或者偏向量

    把b从-0.5加0.5改为0，再计算结果为1>0.5>0更偏离正确值，所以需要减

    把b从-0.5减0.5改为-1，再计算结果为0，输出为0正确

    因为b有变化，需要重新计算第一列，计算结果正确，故b的调整没有问题

  * 第三列、第四列

    输出正确，不用调整权重和偏向量

* 代码实作

  ```python
  import numpy as np
  
  class Perceptron:
      # 初始化输入 输入向量的尺寸（输入的向量包含几个元素）、权重、偏向量
      def __init__(self, input_lenth, weights=None, bias=None):
          if weights is None:
              self.weights = np.ones(input_lenth) * 1
          else:
              self.weights = weights
          if bias is None:
              self.bias = -1
          else:
              self.bias = bias
  
      # 静态方法：纯逻辑方法，与类中属性无关纯放在里面托管，不可以使用self等参数
      @staticmethod
      def activation_function(x):
          if x > 0:
              return 1
          else:
              return 0
  
      # call函数是将类名做成一个同名函数来用
      def __call__(self, input_data):
          weighted_input = self.weights * input_data
          weighted_sum = weighted_input.sum() + self.bias
          return Perceptron.activation_function(weighted_sum)
  
  
  weights = np.array([1, 1])
  bias = -1
  AND_GATE = Perceptron(2, weights, bias)
  
  input_data = [
      np.array([0, 0]),
      np.array([0, 1]),
      np.array([1, 0]),
      np.array([1, 1])
  ]
  
  for x in input_data:
      out = AND_GATE(np.array(x))
      print(x, out)
  ```

  ```python
  [0 0] 0
  [0 1] 0
  [1 0] 0
  [1 1] 1
  ```

#### 感知器范例（二）：OR逻辑元

* 与上类似，权重仍为（1，1），bias改为-0.5即可

#### 线性不可分问题

* 对于**AND**和**OR**，将输出为0画作 □ ，输出为1画作 ○ ，2个输入画作坐标，则一条直线即可区分开这2种不同的输出，而同样的操作对于异或**XOR**来说，（0，0）和（1，1）是0，（1，0）和（0，1）是1，无法只用一条直线区分开，即单一感知器解决不了线性不可分问题，需要多层感知器（即多条直线）
* 需要使用二层感知器来解决**XOR**问题

#### 感知器范例（三）：XOR逻辑元

* 真值表：

  |  x1  |  x2  | out  |
  | :--: | :--: | :--: |
  |  0   |  0   |  0   |
  |  0   |  1   |  1   |
  |  1   |  0   |  1   |
  |  1   |  1   |  0   |

* 首先使用2条函数线可以把平面空间区分成3个部分，成功划分出out的2类：

  * h1（x1，x2）=  x1 + x2 - 0.5
  * h2（x1，x2）=  x1 + x2 - 1.5

* 过程：

  * h1（0，0） = f（1x0+1x0-0.5） = 0
  * h1（0，1） = f（1x0+1x1-0.5） = 1
  * h1（1，0） = f（1x1+1x0-0.5） = 1
  * h1（1，1） = f（1x1+1x1-0.5） = 1
  * h2（0，0） = f（1x0+1x0-1.5） = 0
  * h2（0，1） = f（1x0+1x1-1.5） = 0
  * h2（1，0） = f（1x1+1x0-1.5） = 0
  * h2（1，1） = f（1x1+1x1-1.5） = 1

* 由以上（h1,h2）组成的点分别为（0，0）、（1，1）这2个 □ 和 2个（1，0）画作的 ○ 

  接下来可以由  F（h1，h2）= h1 - 2h2 - 0.5 = 0 来划分这两类点了

  * （x1，x2）=（0，0）->（0，0）代入 = 0
  * （x1，x2）=（0，1）->（1，0）代入 = 1
  * （x1，x2）=（1，0）->（1，0）代入 = 1
  * （x1，x2）=（1，1）->（1，1）代入 = 0

* 代码实作

  ```python
  import numpy as np
  
  class Perceptron:
      def __init__(self, input_lenth, weights=None, bias=None):
          if weights is None:
              self.weights = np.ones(input_lenth) * 1
          else:
              self.weights = weights
          if bias is None:
              self.bias = -1
          else:
              self.bias = bias
  
      @staticmethod
      def activation_function(x):
          if x > 0:
              return 1
          else:
              return 0
  
      def __call__(self, input_data):
          weighted_input = self.weights * input_data
          weighted_sum = weighted_input.sum() + self.bias
          return Perceptron.activation_function(weighted_sum), weighted_sum  #多传回sum值
  
  # 第一条线（感知器）
  weights = np.array([1, 1])
  bias = -0.5
  h1 = Perceptron(2, weights, bias)
  
  # 第二条线
  weights = np.array([1, 1])
  bias = -1.5
  h2 = Perceptron(2, weights, bias)
  
  # 第三条线
  weights = np.array([1, -2])
  bias = -0.5
  F = Perceptron(2, weights, bias)
  
  input_data = [
      np.array([0, 0]),
      np.array([0, 1]),
      np.array([1, 0]),
      np.array([1, 1])
  ]
  
  for x in input_data:
      out1, w1 = h1(np.array(x))
      out2, w2 = h2(np.array(x))
      first_result = np.array([w1, w2])
      new_point = np.array([out1, out2])
      out, w = F(new_point)
      print(x, first_result, new_point, out)
  ```

  ```python
  [0 0] [-0.5 -1.5] [0 0] 0
  [0 1] [ 0.5 -0.5] [1 0] 1
  [1 0] [ 0.5 -0.5] [1 0] 1
  [1 1] [1.5 0.5] [1 1] 0
  ```

#### 多层感知器就是神经网络

* 感知器就是神经元
* 二层**XOR**逻辑单元其中第一层是隐藏层（h1和h2）,第二层是输出层。输入层是x1和x2组成。

#### 隐藏层神经元个数怎么设定

* 每一个隐藏层的神经元数量建议是一致的
* 隐藏层的神经元数量是在输入层和输出层之间，为2/3输入层神经元数量+输出层神经元数
* 隐藏层的神经元数量应少于2倍的输入层神经元数

#### NAND逻辑元

* not and，非与门
* 权重分别为-1，-1，偏向量为1.5

#### 深度学习的学习过程

* 学习目标：找出正确的权重来缩小损失（偏差）

* ​     正向传播 

  → 评估损失（使用损失函数（Loss Function）计算损失分数（Loss Score））

  → 反向传播（使用优化器（Optimizer）更新权重，使用反向传播计算出每一层权重需要分担损失的梯度）

* 一般数据集比较大，每次只使用 批次（Batch）来训练，当整个资料集都经过正向和反向传播阶段的神经网络，即称为一个 训练周期（Epoch）

#### 深度学习学到了什么

* 低度拟合（Undefitting）
* 最佳化    （Optimum）
* 过度拟合（Overfitting）

#### 启动函数 Activation Fuction

* 启动函数是一种非线性函数，可以打破线性关系

* 隐藏层：

  * **ReLU ** 函数：

    * 公式：max（0，x）

    * 大部分微分为1，解决了梯度消失问题

      ```python
      import numpy as np
      import matplotlib.pyplot as plt
      
      def relu(x):
          return np.maximum(0,x)
      
      x = np.arange(-6,6,0.1)
      
      plt.plot(x,sigmoid(x))
      plt.title("sigmoid function")
      plt.show()
      ```

* 输出层：

  * **Sigmoid** 函数：

    * 公式：1/（1+e^-x）

    * 将资料转换为0~1之间的几率，大部分结果非常接近0或1

      ```python
      import numpy as np
      import matplotlib.pyplot as plt
      
      def sigmoid(x):
          return 1/(1+(np.e**(-x)))
      
      x = np.arange(-6,6,0.1)
      
      plt.plot(x,sigmoid(x))
      plt.title("sigmoid function")
      plt.show()
      ```

    * 该函数微分最大为1/4，所以几层反向后梯度迅速为0，一般只在输出层用这个

  * **Tanh** 函数：

    * 公式：sinh（x）/ cosh（x）

    * 双曲三角函数，输出在-1~1之间，优点是可以处理负值

      ```python
      import numpy as np
      import matplotlib.pyplot as plt
      
      def tanh(x):
          return np.tanh(x)
      
      x = np.arange(-6,6,0.1)
      
      plt.plot(x,sigmoid(x))
      plt.title("sigmoid function")
      plt.show()
      ```

  * **Softmax** 函数

    * 将输入值反馈为对应的出现频率

      ```python
      import numpy as np
      
      def softmax(x):
          return np.exp(x)/sum(np.exp(x))
      
      x = np.array([1,2,3,4,1,2,3])
      
      y = softmax(x)
      print()
      ```

      ```python
      [0.02364054 0.06426166 0.1746813  0.474833   0.02364054 0.06426166
       0.1746813 ]
      ```

#### 损失函数 Loss Function

* 回归问题一般用 均方误差（Mean Square Error），分类问题一般用 交叉熵（Cross-Entropy）

* 均方误差：

  * y：输出预设值，t：目标值，y - t 的平方和乘以 1/2

    ```python
    def MSE(y,t):
        return 0.5*np.sum((y-t)**2)
    ```

* 交叉熵：

  * 资讯量：

    * 大小与几率成反比，越不可能发生的事资讯量越大
    * H（x）=  - log2（P）

  * 资讯熵：

    * 量化资讯混乱程度
    * H（x）= Σ [ P (x) * log2（P(x)）]

  * 举例：

    * 64次比赛，A赢了63次，B赢了1次

      A的资讯量：-log2（63/64）= 0.023

      B的资讯量：-log2（1/64）= 6

      资讯熵：0.023 x（63/64）+ 6 x （1/64）= 0.116644

    * 资讯确定，不混乱，资讯熵小：一面倒的情况

    * 资讯不确定，混乱，资讯熵大：五五开

  * 交叉熵：

    * 交叉熵是使用资讯熵来评估2组几率向量之间的差异程度，当交叉熵越小，2组几率向量越接近

    * 公式：

      ​	H（x，y）= - Σ P(x) * log2 P(y)， 其中x为目标值，y为预测值

    * 举例：

      目标值：[1/4，1/4，1/4，1/4]

      所得2组值：

      ​		Y1 = [1/4，1/2，1/8，1/8]

      ​		Y2 = [1/4，1/4，1/8，1/2]

      计算：

      ​		H（X，Y1）= -（ 1/4log2(1/4) + 1/4log2(1/2) + 1/4log2(1/8) +1/4log2(1/8) ）= 2.25

      ​		H（X，Y2）= 2

      所以 Y2 比 Y1 更接近 X

#### 反向传播算法和梯度下降法

* 神经网络使用优化器来更新神经网络的权重，优化器是使用反向传播计算出每一层权重需要分担损失的梯度，然后再使用梯度下降法更新神经网络的每一个神经层的权重
* 注意：梯度下降法找的是一个相对的最佳解（局部最佳），不一定能找到全域最佳解，即函数的最低点

> #### 梯度下降法

* 公式：w1 = w0 - α *（ ∂L(w) / ∂w0 ）

  其中 α 是学习率（步数大小，太小费时，太大可能错过最佳解）

* 单变数函数的梯度下降法实例：

  * 函数：L(w) = w**2 ，w0起点是5，学习率是0.4

  * 函数微分是：2w

  * w0 = 5

    w1 = w0 - 0.4X 5X2 = 1

    w2 = 0.2

    ......

    ```python
    import matplotlib.pyplot as plt
    import numpy as np
    
    def L(w):
        return w * w
    
    def dL(w):
        return 2 * w
    
    def gradient_descent(w_start, df, lr, epochs):
        w_gd = []
        w_gd.append(w_start)
        pre_w = w_start
    
        for i in range(epochs):
            w = pre_w - lr * df(pre_w)
            w_gd.append(w)
            pre_w = w
        return np.array(w_gd)
    
    w0 = 5
    epochs = 5
    lr = 0.4
    w_gd = gradient_descent(w0, dL, lr, epochs)
    print(w_gd)
    
    # 画出图像
    
    t = np.arange(-5.5, 5.5, 0.01)
    plt.plot(t, L(t), c='b')
    plt.plot(w_gd, L(w_gd), c='r', label='lr={}'.format(lr))
    plt.scatter(w_gd, L(w_gd), c='r')
    plt.legend()
    plt.show()
    ```

    

* 多变数函数的梯度下降法实例：

  * 函数 L(w1,w2) = w1^2 + w2^2，w0起点是[2,4]，学习率是0.1

  * 函数的微分是：2w1+2w2

  * w0 = [2，4]

    w1 = w0 - 0.1X[4，8] = [1.6，3.2]

    ......

    ```python
    import numpy as np
    import matplotlib.pyplot as plt
    
    def L(w1, w2):
        return w1**2 + w2**2
    
    def dL(w):
        return np.array([2*w[0], 2*w[1]])
    
    def gradient_descent(w_start, df, lr, epochs):
        w1_gd = []
        w2_gd = []
        w1_gd.append(w_start[0])
        w2_gd.append(w_start[1]) 
        pre_w = w_start
    
        for i in range(epochs):
            w = pre_w - lr*df(pre_w)
            w1_gd.append(w[0])
            w2_gd.append(w[1])
            pre_w = w
    
        return np.array(w1_gd), np.array(w2_gd)
    
    w0 = np.array([2, 4])
    lr = 0.1
    epochs = 40
    
    x1 = np.arange(-5, 5, 0.05)
    x2 = np.arange(-5, 5, 0.05)
    
    w1, w2 = np.meshgrid(x1, x2)
    
    fig1, ax1 = plt.subplots()
    ax1.contour(w1, w2, L(w1, w2), levels=np.logspace(-3, 3, 30), cmap='jet')
    min_point = np.array([0., 0.])
    min_point_ = min_point[:, np.newaxis]
    ax1.plot(*min_point_, L(*min_point_), 'r*', markersize=10)
    ax1.set_xlabel('w1')
    ax1.set_ylabel('w2')
    
    w1_gd, w2_gd = gradient_descent(w0, dL, lr, epochs)
    w_gd = np.column_stack([w1_gd, w2_gd])
    print(w_gd)
    
    ax1.plot(w1_gd, w2_gd, 'bo')
    for i in range(1, epochs+1):
        ax1.annotate('', xy=(w1_gd[i], w2_gd[i]), 
                       xytext=(w1_gd[i-1], w2_gd[i-1]),
                       arrowprops={'arrowstyle': '->', 'color': 'r', 'lw': 1},
                       va='center', ha='center')
    plt.show()
    ```

#### 标签资料的One-hot 编码

```python
import numpy as np

def one_hot_encoding(raw, num):
    result = []
    for ele in raw:
        arr = np.zeros(num)
        np.put(arr, ele, 1)
        result.append(arr)
    return np.array(result)

digits = np.array([1, 8, 5, 4])
one_hot = one_hot_encoding(digits, 10)
print(digits)
print(one_hot)
```

```python
[1 8 5 4]
[[0. 1. 0. 0. 0. 0. 0. 0. 0. 0.]
 [0. 0. 0. 0. 0. 0. 0. 0. 1. 0.]
 [0. 0. 0. 0. 0. 1. 0. 0. 0. 0.]
 [0. 0. 0. 0. 1. 0. 0. 0. 0. 0.]]
```

#### 特征标准化-正规化

* 正规化也称最大最小值缩放
* X = （X-Xmin）/（Xmax-Xmin）

```python
import numpy as np

def normalization(raw):
    max_value = max(raw)
    min_value = min(raw)
    norm = [(float(i) - min_value) / (max_value - min_value) for i in raw]
    return norm

x = np.array([255, 128, 45, 0])
print(x)
norm = normalization(x)
print(norm)
print(x / 255)
```

```python
[255 128  45   0]
[1.0, 0.5019607843137255, 0.17647058823529413, 0.0]
[1.         0.50196078 0.17647059 0.        ]
```

#### 特征标准化-标准化

* 标准化也称 Z分数（Z-score），可以位移成平均值是0，标准差是1的资料分布
* X = （X - 平均值）/ 标准差

```python
import numpy as np
from scipy.stats import zscore

x = np.array([255, 128, 45, 0])
z_score = zscore(x)
print(z_score)
print(zscore([[1, 2, 3], [6, 7, 8]], axis=1))
```

```python
[ 1.52573266  0.21648909 -0.63915828 -1.10306348]
[[-1.22474487  0.          1.22474487]
 [-1.22474487  0.          1.22474487]]
```

#### 反向传播算法-Backpropagation

* 整个神经网络的训练循环事实上就是反向传播算法（BP）
  * 前向传播阶段
  * 反向传播阶段
  * 权重更新阶段
* 计算out对某个权重的微分
