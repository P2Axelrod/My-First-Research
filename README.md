配置环境的命令
```
conda create -n Axelrod python=3.7
pip install -r requirements.txt
```
cuda拓展的编译
```
cd RFCR-Pytorch/KPConv-deform/cpp_wrappers
sh compile_wrappers.sh
```

运行代码的命令
```
python train_S3DIS.py 
```
运行结束后会生成一个results文件夹
