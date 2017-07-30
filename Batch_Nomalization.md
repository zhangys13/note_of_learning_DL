# Batch Nomalization

Udacity 学习笔记，2017.7.28   
[TOC]
##1、What 是什么
### 发明：

Batch normalization（BN）是由Sergey Ioffe 和 Christian Szegedy 在2015年提出的，在 paper [Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift](https://arxiv.org/pdf/1502.03167.pdf) 里。   
###Idea：   
不只在 **input layer** 去做输入归一化，而是在**每一层 layer** 都做输入归一化。   
‘Batch’ 是指在归一化计算时，基于当前的 mini-batch 去计算 mean 和 variance。   
灵感来自于“将输入规范化到网络可以帮助网络学习”，可以想象中间某层，从前层接收的，也是某种程度的“输入”。   
##2、Why 什么好处
总的来说，能帮助更好的 training。减少一些“炼丹”调参的压力。   
### 2.1 网络训练更快
其实，加了 BN，计算量增加的，一个 epoch 需要更多的计算时间。但收敛变快，应该需要的 epoch 数量减少，总的训练时间变短。
### 2.2 允许更高的学习率
Gradient descent一般需要很小的 learning rate，否则容易跑飞，不收敛。随着网络变得 deeper，梯度在 BP 的过程中越来越小。训练需要的 epoch 越来越多。   
BN可以改变这个现状限制（个人感觉和第一点很重叠）。
### 2.3 使权重更容易初始化
网络参数的初始化，之前是很敏感的。网络越 deeper，就越要小心。BN 的应用，使得 Weight initialize 不再那么“敏感”。
### 2.4 使更多的激活函数变得可用
有些激活函数在饱和状态下 do not work well。比如：   
Sigmoid 函数很容易梯度变得很小，这使它很难在深度网络中使用。   
ReLU 可能在训练中进入“死亡”（在负半轴梯度消失），可能完全无法学习。所以必须小心它的 value 变化。   
BN可以修正 Activate Function 的输入值，这使得之前不好用的函数，变得可用。
### 2.5 简化更深层网络的创建
一般认为，更深的网络效果更好，但以前，构建 deeper 的网络是很难的。上面列出的前4项，使用BN， 可以更容易构建和更快地训练更深层次的神经网络。
### 2.6 增加一些泛化能力（Provides a bit of regularlization）
BN在网络中加入了一些噪声。   
在某些 case 中，比如 Inception 中，BN可以起到 dropout 类似的作用。   
通常，可以认为BN能加入一些额外的正则化，减少一些对 dropout 的需求。
### 2.7 总体上可能给出更好的结果
一些测试case表明，BN能提高训练的结果。   
然而，BN应该当做是提高训练速度的优化，你应该想其他办法是网络效果更优。     
BN使构建更深的网络变容易，也很有帮助。
## 3 How 具体如何计算

### 3.1 计算 mean 和 variance
这里被计算平均值的不是 input layer 的输入，而是BN插入点，前面连着的 node 的输出。    
通常，BN 插在某层的激活函数前，那就计算该层 hidden unit 的输出的 mini batch 平均值。    
**（问题：μB 是每个output channel 一个？还是整个 feature map 整体一个？）**   
（回答：都不是。详见4.2节，而是每个输出点，都统计一个 mean 和 variance。公式(1)里的 m 就是 mini batch size。μ、σ、γ、β的不只有一组，而是看 fc_layer 里有多少 hidden units。关于 CNN 的参见 章节5.1）   
$$
\mu_B \leftarrow \frac{1}{m}\sum_{i=1}^m x_i
$$
$$
\sigma_{B}^{2} \leftarrow \frac{1}{m}\sum_{i=1}^m (x_i - \mu_B)^2
$$
### 3.2 归一化计算
$$
\hat{x_i} \leftarrow \frac{x_i - \mu_B}{\sqrt{\sigma_{B}^{2} + \epsilon}}
$$
有一个**超参** $\epsilon$ 可以确保我们不要尝试除以零，也可以为每个批次增加略微的差异（噪声帮助范化）。**通常取 0.001**。   
**为什么增加一点方差？** 这在统计上说得通的。因为，虽然我们是对 mini batch 做归一化，但我们也在试图依据整个 training set 的分布。training set它本身就是将来应用时，整个可能数据集的一个子集。 全集的方差 比 子集的方差大一点，因此每个mini batch增加一点点的方差，也是考虑到这一点。
### 3.3 线性变化（γ 和 β）

把 x 进行归一化后，我们得到了 $\hat{xi}$ 。但我们不是直接把它输出给后一个 node，而是要对它进行线性变化，如下公式。   
γ 和 β 都是可以**在 training 中学习的参数**，就像 weight、bias一样。   
因为它们像weight一样可以学习，所以他们给你的网络提供一些额外的旋转（在 feature 空间里），以帮助网络学习它试图逼近的功能。   
$$
y_i \leftarrow \gamma \hat{x_i} + \beta
$$
### 3.4 输出到 激活函数

一般而言，BN 都是在激活函数前使用的。   
发明 BN 的 paper 里提到，BN 也可以插在激活函数后面，但实际中很难用。

### 3.5 两种不同的 μ 和 σ

（1）另外有 **pop_mean 和 pop_variance**，在 **inference** 时使用。   
他们会统计整个 training set。用来逼近 general population distribution，它们将用于训练完成后的 inference 过程。   
下面公式表示，在新的 第 t+1次 mini batch时，pop_mean如何更新（pop_variance 类似）。decay 常用0.99。
$$
\mu_{pop,t+1}=\mu_{pop,t}*decay+\mu_{batch,t+1}*(1-decay)
$$

（2）**Why** 要用两组不同   
因为，若 inference 一个 sample（即 mini batch size = 1），则平均值 μ 就等于 x_in（只有一个数做平均），归一化后自然等于0。 BN 输出只与β有关，与输入 sample 无关了。   
所以需要估算general population distribution的 μ 和 σ，在 inference 时使用。

## 4、How 基于TensorFlow怎么使用BN

### 4.1 high level 的调用

[tf.layers.batch_normalization](https://www.tensorflow.org/api_docs/python/tf/layers/batch_normalization) 是相对high level 的抽象调用。   
使用大概如下：
`x_hat = tf.layers.batch_normalization(x, training=self.is_training)`

### 4.2 lower level 的调用

[tf.nn.batch_normalization](https://www.tensorflow.org/api_docs/python/tf/nn/batch_normalization) 是 lower level 的调用。
可以看到更多细节，比如，对于一个 fc 层，实现如下。   
可以看出以下几点：
1. 之前 γ, β, the population mean and variance 都是隐藏在 layer 里做好了，现在是显式的使用。   
2. 初始 γ 赋值1，β 赋值零。这样在计算线性变化$y_i \leftarrow \gamma \hat{x_i} + \beta$时，其实是 identity 函数。但随着 BP，γ、β会慢慢变为想要的值。   
3. pop_mean 和 pop_variance是设定为`trainable=False`的，即不由 BP 去优化。而是我们显式的通过`tf.assign`去更新其值。
4. TensorFlow 不会自动的运行`tf.assign`，因为在 graph 中，它没有被连接在 opt node 所能朔回“树”结构里，`tf.assign`是一个孤立点，不被依赖。   
   通过 with 语句，可以增添依赖关系:`with tf.control_dependencies([train_mean, train_variance]):`
5. ​最底层运算仍是隐式的，藏在`tf.nn.batch_normalization`里。
6. 在 training 和 inference 时，使用的是不同的μ和σ。使用 `tf.cond`方法，通过`self.is_training`变量，确定当前状态，来使用不同的μ和σ。
7. 使用`tf.nn.moments`函数，可以计算mini batch 的 μ和σ。

```python
# Batch normalization uses weights as usual, but does NOT add a bias term. This is because    
# its calculations include gamma and beta variables that make the bias term unnecessary.   
weights = tf.Variable(initial_weights)
linear_output = tf.matmul(layer_in, weights)

num_out_nodes = initial_weights.shape[-1]

# Batch normalization adds additional trainable variables: 
# gamma (for scaling) and beta (for shifting).
gamma = tf.Variable(tf.ones([num_out_nodes]))
beta = tf.Variable(tf.zeros([num_out_nodes]))

# These variables will store the mean and variance for this layer over the entire training set,
# which we assume represents the general population distribution.
# By setting `trainable=False`, we tell TensorFlow not to modify these variables during
# back propagation. Instead, we will assign values to these variables ourselves. 
pop_mean = tf.Variable(tf.zeros([num_out_nodes]), trainable=False)
pop_variance = tf.Variable(tf.ones([num_out_nodes]), trainable=False)

# Batch normalization requires a small constant epsilon, used to ensure we don't divide by zero.
# This is the default value TensorFlow uses.
epsilon = 1e-3

def batch_norm_training():
    # Calculate the mean and variance for the data coming out of this layer's linear-combination step.
    # The [0] defines an array of axes to calculate over.
    batch_mean, batch_variance = tf.nn.moments(linear_output, [0])

    # Calculate a moving average of the training data's mean and variance while training.
    # These will be used during inference.
    # Decay should be some number less than 1. tf.layers.batch_normalization uses the parameter
    # "momentum" to accomplish this and defaults it to 0.99
    decay = 0.99
    train_mean = tf.assign(pop_mean, pop_mean * decay + batch_mean * (1 - decay))
    train_variance = tf.assign(pop_variance, pop_variance * decay + batch_variance * (1 - decay))

    # The 'tf.control_dependencies' context tells TensorFlow it must calculate 'train_mean' 
    # and 'train_variance' before it calculates the 'tf.nn.batch_normalization' layer.
    # This is necessary because the those two operations are not actually in the graph
    # connecting the linear_output and batch_normalization layers, 
    # so TensorFlow would otherwise just skip them.
    with tf.control_dependencies([train_mean, train_variance]):
        return tf.nn.batch_normalization(linear_output, batch_mean, batch_variance, beta, gamma, epsilon)

def batch_norm_inference():
    # During inference, use the our estimated population mean and variance to normalize the layer
    return tf.nn.batch_normalization(linear_output, pop_mean, pop_variance, beta, gamma, epsilon)

# Use `tf.cond` as a sort of if-check. When self.is_training is True, TensorFlow will execute 
# the operation returned from `batch_norm_training`; otherwise it will execute the graph
# operation returned from `batch_norm_inference`.
batch_normalized_output = tf.cond(self.is_training, batch_norm_training, batch_norm_inference)

# Pass the batch-normalized layer output through the activation function.
# The literature states there may be cases where you want to perform the batch normalization *after*
# the activation function, but it is difficult to find any uses of that in practice.
return activation_fn(batch_normalized_output) 
```
### 4.3 更底层的实现
之前在函数`batch_norm_training()`里，使用了如下语句：
```python
return tf.nn.batch_normalization(linear_output, batch_mean, batch_variance, beta, gamma, epsilon)
```
它可以再拆分为以下的 code：
```python
normalized_linear_output = (linear_output - batch_mean) / tf.sqrt(batch_variance + epsilon)
return gamma * normalized_linear_output + beta
```
## 5、在FC 以外的其他NN 中使用

### 5.1 CNN

（1）定义明确一下：   
一个 CNN layer 输出若干个 feature_map。   
一个 feature_map，对应一个 CNN kernel，shape 为 o_h\*o_w\*1 。   

（2）一组 μ 和 σ 对应一个 feature map，而不是之前 fc_layer 里的一个 hidden unit。   
对应代码，之前是：

```python
batch_mean, batch_variance = tf.nn.moments(linear_output, [0])
```
现在变为：
```python
batch_mean, batch_variance = tf.nn.moments(conv_layer, [0,1,2], keep_dims=False)
```
第二个参数, [0,1,2], 告诉TensorFlow 每个feature map计算一组 μ 和 σ。 (这三个轴分别是 batch, height, width)。    
设定`keep_dims=False` 是告诉tf.nn.moments返回的 size 不需要与 input 一致。
### 5.2 RNN
BNN 也可以用在 RNN 里。 可以参考这篇2016的 paper [Recurrent Batch Normalization](https://arxiv.org/abs/1603.09025). 实现起来需要一些工作量。主要是 **一组 μ 和 σ 是基于每个 time step，而不是每个 layer。**   
You can find an example where someone extended `tf.nn.rnn_cell.RNNCell` to include batch normalization in [this GitHub repo](https://gist.github.com/spitis/27ab7d2a30bbaf5ef431b4a02194ac60).