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