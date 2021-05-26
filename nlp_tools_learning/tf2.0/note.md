# 基础知识
## 类型

通过tf.constant()可以创建3中数据类型，分别是数值、布尔、字符串

```
# 标量
tf.constant(2., dtype=tf.float16)

# 向量
tf.constant([2,3], dtype=tf.int16)

# 张量 维度>2
tf.constant([[[1,2], [3,4]], [[5,6], [7,8]]])
```

## 数值精度

1. tf.float16
2. tf.float32
3. tf.float64 也是tf.double
4. tf.int16
5. tf.int32
6. tf.int64

### 类型转换
进行类型转换时，需要保证转换的合法性，例如从高精度转换为低精度时，可能会出现数据溢出

```
a = tf.constant(123456789, dtype=tf.int32) tf.cast(a, tf.int16) # 转换为低精度整型

<tf.Tensor: id=38, shape=(), dtype=int16, numpy=-13035>
```

## 待优化的张量
有些数值型数据需要计算张量，而有些不需要。TensorFlow增加了一种专门的数据类型来记录支持梯度信息的数据：tf.Variable()，包含name，trainable等属性来支持计算图的构建。

```
a = tf.Variable(12, name='hahatest')
a.name
    'hahatest:0'
a.trainable
    True
```
其中trainble属性用来表示当前数据是否需要被优化，默认值是True，也可设置成False来确保该数据不被优化。

## 创建张量
在TensorFlow中有多种方式创建张量：

1. python列表
2. numpy数组
3. 采样自某种已知分布

### 从numpy数组或python列表等容器中创建
```
tf.convert_to_tensor([1, 2.])
<tf.Tensor: shape=(2,), dtype=float32, numpy=array([1., 2.], dtype=float32)>


tf.convert_to_tensor(np.array([[1, 2.], [3,4]]))
<tf.Tensor: shape=(2, 2), dtype=float64, numpy=
array([[1., 2.],
       [3., 4.]])>
```
**numpy的浮点数默认是64位**

### 创建全0/全1张量
```
tf.zeros([2,2])
<tf.Tensor: shape=(2, 2), dtype=float32, numpy=
array([[0., 0.],
       [0., 0.]], dtype=float32)>


tf.ones([2,3])
<tf.Tensor: shape=(2, 3), dtype=float32, numpy=
array([[1., 1., 1.],
       [1., 1., 1.]], dtype=float32)>
```

还可一通过tf.zeros_like()，创建与某张量shape一致的全0张量。
tf.ones_like()同理。

### 自定义数值张量
实际上除了初始化全0，全1向量外，我们偶尔也会用到其他数值的向量，TensorFlow同样提供了函数快速创建, tf.fill()。
```
tf.fill([2, 3], -1)
<tf.Tensor: shape=(2, 3), dtype=int32, numpy=
array([[-1, -1, -1],
       [-1, -1, -1]], dtype=int32)>
```



