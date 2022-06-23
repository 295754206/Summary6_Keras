#### Keras预建神经层类型

* Sequential模型：一种线性堆叠结构，神经层单一输入和单一输出，每一层链接下一层，不允许跨层连接。

* Keras的Sequential模型是一个容器，可以将各种Keras预建神经层类型依序新增至模型中：

  * 多层感知器 MLP：
    * 新增1至多个Dense层后使用Dropout层防止过度拟合
  * 卷积神经网络 CNN：
    * 依序新增1至多组Conv2D和Pooling层后使用Dropout、Flatten和Dense层

  * 循环神经网络 RNN：
    * 分别使用SimpleRNN、LSTM或GRU来建立循环神经网络

#### 模型建立的顺序

* 资料预处理 → 定义模型 → 编译模型 → 训练模型 → 评估模型

#### 编译模型时所使用的函数

| 问题种类        | 输出层启动函数 | 损失函数                 |
| :-------------- | -------------- | :----------------------- |
| 二元分类        | sigmiod        | binary_crossentropy      |
| 单标签多元分类  | softmax        | categorical_crossentropy |
| 多标签多元分类  | sigmoid        | binary_crossentropy      |
| 回归分析        | 不需要         | mse                      |
| 回归值在0~1之间 | sigmoid        | mse或binary_crossentropy |

二元分类是指分成2类，多元分类是分成多个种类，单标签是指是指只属于一类，多标签是指可以属于多类

#### 多层感知器怎么提高模型效能（或调整）

* 特征标准化
* 在输出层使用softmax启动函数
* 在神经层使用权重初始器
* 在编译模型使用adam优化器
* 减少神经网络的参数量