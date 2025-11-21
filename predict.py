# from ultralytics.yolo.utils.configfile import __predict
# from ultralytics import YOLO
engineresults
if pred_boxes and show_boxes:

# def predictmodel(predictConfig,sourceConfig):
      开始加载模型
      # model = YOLO(__predict(predictConfig))
      指定训练参数开始测试
      # for i in model.predict(source=sourceConfig, stream=True, conf=0.15, iou=0.55,
                             # project="runs/predict", name='exp', save_txt=True, save=True):
            # print(i)

# if __name__ == "__main__":
      填写测试的网络模型名称
      # predictConfig = "runs/train/exp4/weights/best.pt"

      填写测试图片文件夹
      # sourceConfig = 'datasets/yolo_format/VisDrone2019-DET-test-dev/images'

      调用测试方法
      # predictmodel(predictConfig,sourceConfig)

from ultralytics import YOLO
import argparse
import os

def predictmodel(predictConfig, sourceConfig, output_dir="runs/predict", name="exp", 
                conf=0.15, iou=0.55, save_txt=True, save=True):
    """批量预测并获取结果"""
    # 加载模型
    model = YOLO(predictConfig)
    
    # 批量预测
    results = model.predict(
        source=sourceConfig,
        stream=True,
        conf=conf,
        iou=iou,
        project=output_dir,
        name=name,
        save_txt=save_txt,
        save=save
    )
    
    # 统计结果
    total_images = 0
    total_detections = 0
    
    for result in results:
        total_images += 1
        detections = len(result.boxes) if result.boxes is not None else 0
        total_detections += detections
        print(f"图像 {total_images}: 检测到 {detections} 个目标 - {result.path}")
    
    print(f"\n批量预测完成！")
    print(f"处理图像数量: {total_images}")
    print(f"总检测目标数: {total_detections}")
    print(f"结果保存在: {output_dir}/{name}")

if __name__ == "__main__":
    # 设置参数
    predictConfig = "runs/train/mmff/weights/best.pt"
    sourceConfig = 'datasets/yolo_format/VisDrone2019-DET-test-dev/images'
    output_dir = "runs/predict"
    name = "exp"
    
    # 调用测试方法
    predictmodel(predictConfig, sourceConfig, output_dir, name)