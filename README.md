配置环境的命令
```
conda create -n Axelrod python=3.7
conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch
pip install -r requirements.txt
conda activate Axelrod
```
cuda拓展的编译
```
cd RFCR-KPConv/cpp_wrappers
sh compile_wrappers.sh
```
下载Data.zip放在与train_S3DIS.py同一级目录下
运行代码的命令
```
unzip Data.zip
python train_S3DIS.py 
```
运行结束后会生成一个results文件夹
