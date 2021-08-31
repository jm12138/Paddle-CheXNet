# Paddle-CheXNet
## 一、简介
* 本项目基于 Paddle 框架复现 CheXNet 模型
* 转换并对齐参考项目提供的预训练模型的参数和精度表现]
* 使用本项目重新训练模型，在精度表现上（Avg AUROC 84.7）略优于论文中展示的指标（Avg AUROC 84.1）
* 论文：[CheXNet: Radiologist-Level Pneumonia Detection on Chest X-Rays with Deep Learning](https://arxiv.org/pdf/1711.05225.pdf)
* 参考项目：[arnoweng/CheXNet](https://github.com/arnoweng/CheXNet)
* AIStudio 项目：[论文复现：基于 Paddle2.0 复现 CheXNet 模型](https://aistudio.baidu.com/aistudio/projectdetail/2264427)


## 二、复现精度
* 测试精度：

    ```python
    The final AUROCs: 
    The average AUROC is 0.847
    The AUROC of Atelectasis is 0.8292392176490241
    The AUROC of Cardiomegaly is 0.9142859299652238
    The AUROC of Effusion is 0.8875584024095078
    The AUROC of Infiltration is 0.711814559849058
    The AUROC of Mass is 0.8647347811493854
    The AUROC of Nodule is 0.7921467608091082
    The AUROC of Pneumonia is 0.7684077120089262
    The AUROC of Pneumothorax is 0.8770951989569952
    The AUROC of Consolidation is 0.8160893266094902
    The AUROC of Edema is 0.8986931866913855
    The AUROC of Emphysema is 0.9302391831183919
    The AUROC of Fibrosis is 0.837411708221408
    The AUROC of Pleural_Thickening is 0.7962435140282585
    The AUROC of Hernia is 0.9311000806021127
    ```
 
* 精度对比：

    |     Pathology      | [Wang et al.](https://arxiv.org/abs/1705.02315) | [Yao et al.](https://arxiv.org/abs/1710.10501) | [CheXNet](https://arxiv.org/abs/1711.05225) | [arnoweng/CheXNet](https://github.com/arnoweng/CheXNet) Release Model  | [arnoweng/CheXNet](https://github.com/arnoweng/CheXNet) Improved Model | Paddle-CheXNet | 
    | :----------------: | :--------------------------------------: | :--------------------------------------: | :--------------------------------------: | :---------------------: | :----------------: | :----------------: |
    |    Atelectasis     |                  0.716                   |                  0.772                   |                  0.8094                  |         0.8294          |       0.8311       | 0.8292 |
    |    Cardiomegaly    |                  0.807                   |                  0.904                   |                  0.9248                  |         0.9165          |       0.9220       | 0.9143 |
    |      Effusion      |                  0.784                   |                  0.859                   |                  0.8638                  |         0.8870          |       0.8891       | 0.8876 |
    |    Infiltration    |                  0.609                   |                  0.695                   |                  0.7345                  |         0.7143          |       0.7146       | 0.7118 |
    |        Mass        |                  0.706                   |                  0.792                   |                  0.8676                  |         0.8597          |       0.8627       | 0.8647 |
    |       Nodule       |                  0.671                   |                  0.717                   |                  0.7802                  |         0.7873          |       0.7883       | 0.7921 |
    |     Pneumonia      |                  0.633                   |                  0.713                   |                  0.7680                  |         0.7745          |       0.7820       | 0.7684 |
    |    Pneumothorax    |                  0.806                   |                  0.841                   |                  0.8887                  |         0.8726          |       0.8844       | 0.8771 |
    |   Consolidation    |                  0.708                   |                  0.788                   |                  0.7901                  |         0.8142          |       0.8148       | 0.8161 |
    |       Edema        |                  0.835                   |                  0.882                   |                  0.8878                  |         0.8932          |       0.8992       | 0.8987 |
    |     Emphysema      |                  0.815                   |                  0.829                   |                  0.9371                  |         0.9254          |       0.9343       | 0.9302 |
    |      Fibrosis      |                  0.769                   |                  0.767                   |                  0.8047                  |         0.8304          |       0.8385       | 0.8374 |
    | Pleural Thickening |                  0.708                   |                  0.765                   |                  0.8062                  |         0.7831          |       0.7914       | 0.7962 |
    |       Hernia       |                  0.767                   |                  0.914                   |                  0.9164                  |         0.9104          |       0.9206       | 0.9311 |
    | Avg AUROCs | | | 0.841 | 0.843 | 0.848 | 0.847 |

## 三、数据集
* 项目使用的数据集为 dataset
* ChestX-ray 数据集包含 30,805 名患者的 112,120 张正面视图的X射线图像，以及利用 NLP 从相关放射学报告挖掘的 14 类疾病的图像标签（每个图像可以有多个标签）。
* 数据集含有 14 类常见的胸部病理，包括肺不张、变实、浸润、气胸、水肿、肺气肿、纤维变性、积液、肺炎、胸膜增厚、心脏肥大、结节、肿块和疝气
* 下载链接：[dataset dataset](https://nihcc.app.box.com/v/ChestXray-NIHCC/folder/37178474737)

## 四、环境依赖
* 本项目依赖如下模块：

    ```python
    scikit-learn
    paddlepaddle-gpu
    opencv-python
    numpy
    pillow
    ```
* 可通过如下命令安装依赖：

    ```shell
    $ pip install -r requirements.txt
    ```

## 五、快速使用
* 同步项目代码

* 下载数据集并解压至 dataset 文件夹

* 使用如下命令进行模型训练：

    ```
    $ python train.py \
        --data_dir=dataset/images \
        --train_list=dataset/labels/train_list.txt \
        --val_list=dataset/labels/val_list.txt \
        --save_dir=save \
        --batch_size=128 \
        --learning_rate=0.001 \
        --decay_epochs=10,15,18 \
        --decay_factor=0.1 \
        --epoch=20
    ```
        Epoch 1/20
        step  10/614 - loss: 0.1650 - AUROC_Atelectasis: 0.5343 - AUROC_Cardiomegaly: 0.5442 - AUROC_Effusion: 0.5442 - AUROC_Infiltration: 0.5400 - AUROC_Mass: 0.4854 - AUROC_Nodule: 0.5043 - AUROC_Pneumonia: 0.5403 - AUROC_Pneumothorax: 0.5507 - AUROC_Consolidation: 0.5129 - AUROC_Edema: 0.5020 - AUROC_Emphysema: 0.5979 - AUROC_Fibrosis: 0.5571 - AUROC_Pleural_Thickening: 0.5663 - AUROC_Hernia: 0.5823 - AUROC_avg: 0.5401 - 3s/step
        step  20/614 - loss: 0.1895 - AUROC_Atelectasis: 0.5719 - AUROC_Cardiomegaly: 0.5691 - AUROC_Effusion: 0.6262 - AUROC_Infiltration: 0.5605 - AUROC_Mass: 0.5333 - AUROC_Nodule: 0.5022 - AUROC_Pneumonia: 0.5410 - AUROC_Pneumothorax: 0.5947 - AUROC_Consolidation: 0.5511 - AUROC_Edema: 0.5663 - AUROC_Emphysema: 0.5629 - AUROC_Fibrosis: 0.4929 - AUROC_Pleural_Thickening: 0.5867 - AUROC_Hernia: 0.6170 - AUROC_avg: 0.5626 - 3s/step
        step  30/614 - loss: 0.1637 - AUROC_Atelectasis: 0.5881 - AUROC_Cardiomegaly: 0.5548 - AUROC_Effusion: 0.6691 - AUROC_Infiltration: 0.5767 - AUROC_Mass: 0.5438 - AUROC_Nodule: 0.5179 - AUROC_Pneumonia: 0.5331 - AUROC_Pneumothorax: 0.6317 - AUROC_Consolidation: 0.5818 - AUROC_Edema: 0.6350 - AUROC_Emphysema: 0.5629 - AUROC_Fibrosis: 0.5629 - AUROC_Pleural_Thickening: 0.6101 - AUROC_Hernia: 0.7026 - AUROC_avg: 0.5908 - 3s/step
        ...

* 使用如下命令进行模型精度测试（默认使用本项目训练的最佳模型参数）：

    ```
    $ python eval.py \
        --data_dir=dataset/images \
        --test_list=dataset/labels/test_list.txt \
        --batch_size=128 \
        --ckpt=pretrained_models/model_paddle.pdparams
    ```
        => loading checkpoint
        => loaded checkpoint
        100%|█████████████████████████████████████████| 176/176 [12:56<00:00,  4.41s/it]
        The final AUROCs: 
        The average AUROC is 0.847
        The AUROC of Atelectasis is 0.8292392176490241
        The AUROC of Cardiomegaly is 0.9142859299652238
        The AUROC of Effusion is 0.8875584024095078
        The AUROC of Infiltration is 0.711814559849058
        The AUROC of Mass is 0.8647347811493854
        The AUROC of Nodule is 0.7921467608091082
        The AUROC of Pneumonia is 0.7684077120089262
        The AUROC of Pneumothorax is 0.8770951989569952
        The AUROC of Consolidation is 0.8160893266094902
        The AUROC of Edema is 0.8986931866913855
        The AUROC of Emphysema is 0.9302391831183919
        The AUROC of Fibrosis is 0.837411708221408
        The AUROC of Pleural_Thickening is 0.7962435140282585
        The AUROC of Hernia is 0.9311000806021127

## 六、代码结构与详细说明
* 代码结构

    ```python
    │  train.py # 模型训练脚本
    │  eval.py # 模型测试脚本
    │  requirements.txt # 依赖环境列表
    │
    ├─chexnet # ChexNet 代码
    │      data.py # 数据处理
    │      densenet.py # DenseNet
    │      model.py # CheXNet Model
    │      utility.py # 功能代码
    │
    ├─dataset
    │  ├─images # 数据集图像
    │  │
    │  └─labels # 数据集列表
    │          test_list.txt # 测试集
    │          train_list.txt # 训练集
    │          val_list.txt # 验证集
    │
    ├─logs # 训练 log
    │
    └─pretrained_models # 预训练模型
            model_paddle.pdparams # 本项目训练的参数文件
            model_torch.pdparams # 转换自参考项目的参数文件
    ```

* 参数说明：

    |参数|默认值|说明|适用脚本|
    |:-:|:-:|:-:|:-:|
    |data_dir|dataset/images|数据集图片目录|train / eval|
    |save_dir|save|保存目录|train|
    |train_list|dataset/labels/train_list.txt|数据集训练集列表|train|
    |val_list|dataset/labels/val_list.txt|数据集验证集列表|train|
    |test_list|dataset/labels/test_list.txt|数据集测试集列表|eval|
    |batch_size|128|数据处理批大小|train / eval|
    |epoch|20|训练轮次|train|
    |learning_rate|0.001|学习率|train|
    |decay_epochs|10,15,18|学习率衰减轮次|train|
    |decay_factor|0.1|学习率衰减因子|train|
    |ckpt|pretrained_models/model_paddle.pdparams|预训练模型路径|eval|

## 七、模型信息
* 模型的总体信息如下：

    |信息|说明|
    |:-:|:-:|
    |框架版本|Paddle 2.1.2|
    |骨干网络|DenseNet 121|
    |应用场景|多标签分类（胸部 X 光片肺炎检测）|
    |支持硬件| GPU / CPU |

* 具体的网络结构如下图：

    ![](https://ai-studio-static-online.cdn.bcebos.com/864ca7dad49d4a88988675dc791ad465c104a0876d1f460996059283fd2dc467)
