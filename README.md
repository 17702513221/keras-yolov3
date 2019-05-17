## 安装
## keras搭建(目前训练还是在darknet上进行，推理通过keras，相关代码我做了标注，不过不如darknet原版，我更多的是为未来模型优化做的技术积累)
* sudo apt install python3-pip
* pip3 install tensorflow-gpu==1.9.0 -i https://pypi.tuna.tsinghua.edu.cn/simple
* pip3 install keras -i https://pypi.tuna.tsinghua.edu.cn/simple
* pip3 install matplotlib -i https://pypi.tuna.tsinghua.edu.cn/simple
* pip3 install opencv-python -i https://pypi.tuna.tsinghua.edu.cn/simple
## 下载
* wget https://github.com/intel-iot-devkit/sample-videos/raw/master/worker-zone-detection.mp4
* wget https://pjreddie.com/media/files/yolov3.weights
* wget https://pjreddie.com/media/files/darknet53.conv.74
## 模型转换
* python3 convert.py yolov3.cfg yolov3.weights model_data/yolo.h5
## 推理测试
* python3 yolo_video.py --input worker-zone-detection.mp4
## 下载数据集及处理
* wget https://pjreddie.com/media/files/VOCtrainval_06-Nov-2007.tar
* wget https://pjreddie.com/media/files/VOCtest_06-Nov-2007.tar
* tar -xvf VOCtrainval_06-Nov-2007.tar
* tar -xvf VOCtest_06-Nov-2007.tar
* python voc_annotation.py
## 训练
* python train.py
