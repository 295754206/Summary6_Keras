#### 独自Python环境下需要加装的套件（相对于Anaconda）

* numpy
* pandas
* matplotlib

 Python现自带numpy，需另安装tensorflow等套件

```shell
> pip install tensorflow
> pip install matplotlib
> pip install pandas
> pip install scipy
```

#### Anaconda下打开Anaconda Prompt需要加装：

* pip install tensorflow
* pip install keras
* 如果显示numpy套件错误信息需要更新：pip install --upgrade numpy

#### 在Anaconda Prompt下针对不同的Python项目建立专属开发环境

* 建立名字为keras的环境
  * （base）>  conda create --name keras anaconda
* 要求使用特定Python版本
  * （base）>  conda create -name keras36 python=3.6 anaconda
* 显示已建立的环境
  * （base）>  conda env list
* 启用建立的虚拟环境
  * （base）>  activate keras
* 检查该环境下已安装的套件
  * （keras）>  conda list
* 在环境下安装新的套件
  * （keras）>  pip install tensorflow
* 关闭环境
  * （keras）>  deactivate 
* 删除环境
  * （base）>  conda env remove --name keras

#### Pycharm快捷键

* 格式化代码块：alt + ctrl + L
