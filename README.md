- 代码运行在上次的conda虚拟环境
- 数据下载网址：https://drive.google.com/drive/folders/12wLblskNVBUeryt1xaJTQlIoJac2WehV
```
mkdir data
```
下载s3dis.tar.gz放在data目录下
```
tar -zxvf s3dis.tar.gz
```
 运行代码的命令
```
conda activate Axelrod
pip install hydra-core --upgrade
pip install omegaconf
python train_semseg.py
```
训练结束后生成log，下面保存best_model.pth