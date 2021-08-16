# Paddle-CheXNet
## 简介
* 本项目基于 Paddle 框架复现 CheXNet 模型
* 转换并对齐参考项目提供的预训练模型的参数和精度表现
* 使用本项目重新训练模型，在精度表现上（Avg AUROC 84.7）略优于论文中展示的指标（Avg AUROC 84.1）

## 相关资料
* 论文：[CheXNet: Radiologist-Level Pneumonia Detection on Chest X-Rays with Deep Learning](https://arxiv.org/pdf/1711.05225.pdf)

* 数据集：[ChestX-ray14 dataset](https://nihcc.app.box.com/v/ChestXray-NIHCC/folder/37178474737)

* 参考项目：[arnoweng/CheXNet](https://github.com/arnoweng/CheXNet)

* 验收标准：

    ![](https://ai-studio-static-online.cdn.bcebos.com/5a72294f67f74fdfac3cceced5eb767ca6d75a41d41045608433e6be40a04502)

* 精度指标对比：

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

* AIStudio 项目：[论文复现：基于 Paddle2.0 复现 CheXNet 模型](https://aistudio.baidu.com/aistudio/projectdetail/2264427)（包含模型的训练 log 和 ckpt）

## 预训练模型
* 本项目提供了两个预训练模型（放置于 pretrained_models 目录下）：

    * model_from_torch.pdparams -> 转换至 [arnoweng/CheXNet](https://github.com/arnoweng/CheXNet) 项目提供的预训练模型，精度对齐，Avg AUROC 84.1

    * best_model_via_this_project.pdparams -> 基于本项目重新训练的模型，具体训练过程请参考 AIStudio 项目，Avg AUROC 84.7

## 训练过程
* 模型的训练日志放置于 logs 目录中

## 代码使用
* 同步项目代码

* 下载数据集并解压至 ChestX-ray14 文件夹

* 使用如下命令进行模型训练：

    ```
    $ python train.py
    ```
        Epoch 1/20
        step  10/614 - loss: 0.1650 - AUROC_Atelectasis: 0.5343 - AUROC_Cardiomegaly: 0.5442 - AUROC_Effusion: 0.5442 - AUROC_Infiltration: 0.5400 - AUROC_Mass: 0.4854 - AUROC_Nodule: 0.5043 - AUROC_Pneumonia: 0.5403 - AUROC_Pneumothorax: 0.5507 - AUROC_Consolidation: 0.5129 - AUROC_Edema: 0.5020 - AUROC_Emphysema: 0.5979 - AUROC_Fibrosis: 0.5571 - AUROC_Pleural_Thickening: 0.5663 - AUROC_Hernia: 0.5823 - AUROC_avg: 0.5401 - 3s/step
        step  20/614 - loss: 0.1895 - AUROC_Atelectasis: 0.5719 - AUROC_Cardiomegaly: 0.5691 - AUROC_Effusion: 0.6262 - AUROC_Infiltration: 0.5605 - AUROC_Mass: 0.5333 - AUROC_Nodule: 0.5022 - AUROC_Pneumonia: 0.5410 - AUROC_Pneumothorax: 0.5947 - AUROC_Consolidation: 0.5511 - AUROC_Edema: 0.5663 - AUROC_Emphysema: 0.5629 - AUROC_Fibrosis: 0.4929 - AUROC_Pleural_Thickening: 0.5867 - AUROC_Hernia: 0.6170 - AUROC_avg: 0.5626 - 3s/step
        step  30/614 - loss: 0.1637 - AUROC_Atelectasis: 0.5881 - AUROC_Cardiomegaly: 0.5548 - AUROC_Effusion: 0.6691 - AUROC_Infiltration: 0.5767 - AUROC_Mass: 0.5438 - AUROC_Nodule: 0.5179 - AUROC_Pneumonia: 0.5331 - AUROC_Pneumothorax: 0.6317 - AUROC_Consolidation: 0.5818 - AUROC_Edema: 0.6350 - AUROC_Emphysema: 0.5629 - AUROC_Fibrosis: 0.5629 - AUROC_Pleural_Thickening: 0.6101 - AUROC_Hernia: 0.7026 - AUROC_avg: 0.5908 - 3s/step
        ...

* 使用如下命令进行模型精度测试（默认使用本项目训练的最佳模型参数）：

    ```
    $ python eval.py
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