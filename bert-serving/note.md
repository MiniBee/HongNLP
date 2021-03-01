

## 1.下载bert-serving服务
pip install bert-serving-server
pip install bert-serving-client

## 2.下载模型
github.com/google-research/bert

## 3.启动服务
bert-serving-start -model_dir /data/phy/bert/chinese_L-12_H-768_A-12 -num_worker=5

## 4.词向量服务
bert-serving-start -pooling_strategy NONE -model_dir /data/phy/bert/chinese_L-12_H-768_A-12

## ⚠️注意，要用1.0
pip uninstall tensorflow

pip uninstall tensorflow-estimator

and then

pip install tensorflow==1.13.1


## 更新到最新tensorflow
pip install --upgrade tensorflow