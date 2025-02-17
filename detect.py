# -*- coding: utf-8 -*-
"""
@Auth ： 挂科边缘
@File ：detect.py
@IDE ：PyCharm
@Motto:学习新思想，争做新青年
@Email ：179958974@qq.com
"""

from ultralytics import YOLO

if __name__ == '__main__':

    # Load a model
    model = YOLO(model=r'/home/mzc/Yolov11/ultralytics-8.3.74/runs/train/exp/weights/best.pt')
    model.predict(source=r'/home/mzc/Yolov11/ultralytics-8.3.74/images/val',
                  save=True,
                  show=True,
                  )
