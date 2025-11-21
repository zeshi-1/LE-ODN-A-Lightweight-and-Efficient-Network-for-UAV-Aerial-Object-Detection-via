from ultralytics import YOLO
from ultralytics.yolo.utils.configfile import __train
import warnings
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
warnings.filterwarnings('ignore')

# 导入多模态模块
from modules import TextEncoder, C2f_MMFF

class MultiModalYOLO(nn.Module):
    """多模态YOLO模型"""
    def __init__(self, cfg, weights=None):
        super(MultiModalYOLO, self).__init__()
        self.text_encoder = TextEncoder(text_dim=256)
        
        # 加载YOLO模型
        self.yolo_model = YOLO(cfg)
        self.model = self.yolo_model.model  # 获取实际的PyTorch模型
        
        # 替换第一个C2f模块为C2f_MMFF
        self._replace_c2f_with_mmff()
        
        if weights and ".pt" in weights:
            print("+++++++载入预训练权重：", weights, "++++++++")
            self.load_weights(weights)
        else:
            print("-------没有载入预训练权重-------")
    
    def _replace_c2f_with_mmff(self):
        """将backbone中的第一个C2f模块替换为C2f_MMFF"""
        backbone = self.model.model
        for i, module in enumerate(backbone):
            if hasattr(module, '__class__') and module.__class__.__name__ == 'C2f':
                # 获取原C2f模块的参数
                c1 = module.cv1.conv.in_channels
                c2 = module.cv2.conv.out_channels
                n = len(module.m)
                shortcut = module.m[0].add if len(module.m) > 0 else False
                
                # 创建多模态C2f模块
                mmff_module = C2f_MMFF(c1, c2, n, shortcut, text_dim=256, dropout_prob=0.5)
                
                # 替换模块
                backbone[i] = mmff_module
                print(f"已将第{i}个C2f模块替换为C2f_MMFF")
                break
    
    def load_weights(self, weights_path):
        """加载预训练权重，跳过不匹配的层"""
        checkpoint = torch.load(weights_path, map_location='cpu')
        
        if 'model' in checkpoint:
            state_dict = checkpoint['model']
        else:
            state_dict = checkpoint
        
        # 过滤掉不匹配的权重
        model_state_dict = self.model.state_dict()
        filtered_state_dict = {}
        
        for k, v in state_dict.items():
            if k in model_state_dict:
                if v.shape == model_state_dict[k].shape:
                    filtered_state_dict[k] = v
                else:
                    print(f"跳过权重 {k}, 形状不匹配: {v.shape} vs {model_state_dict[k].shape}")
            else:
                print(f"跳过不存在的键: {k}")
        
        # 加载权重
        model_state_dict.update(filtered_state_dict)
        self.model.load_state_dict(model_state_dict, strict=False)
    
    def preprocess_annotations(self, annotations, batch_size):
        """预处理annotation文本"""
        processed_annotations = []
        
        for i in range(batch_size):
            if i < len(annotations) and annotations[i] is not None:
                # 如果有真实的annotation，使用真实文本
                processed_annotations.append(annotations[i])
            else:
                # 如果没有annotation，使用默认文本（表示没有目标）
                processed_annotations.append("no objects in image")
        
        return processed_annotations
    
    def extract_text_features(self, annotations, training=False):
        """提取文本特征"""
        if annotations is None:
            # 如果没有annotation，创建默认文本
            batch_size = 1 if not hasattr(self, 'current_batch_size') else self.current_batch_size
            annotations = ["no objects in image"] * batch_size
        
        # 编码文本
        text_features = self.text_encoder(annotations)
        
        # 训练时随机遮挡文本特征
        if training and torch.rand(1) < 0.5:
            text_features = torch.zeros_like(text_features)
        
        return text_features
    
    def forward(self, x, annotations=None, training=False):
        """
        x: 输入图像 [B, C, H, W]
        annotations: annotation文本列表，长度为B
        training: 是否为训练模式
        """
        B = x.shape[0]
        self.current_batch_size = B
        
        # 预处理annotation
        processed_annotations = self.preprocess_annotations(annotations, B)
        
        # 提取文本特征
        text_features = self.extract_text_features(processed_annotations, training)
        
        # 获取YOLO模型的backbone和head
        backbone = self.model.model
        head = self.model.model[backbone[-1].i] if hasattr(backbone[-1], 'i') else None
        
        # 多模态前向传播
        y = []
        for i, module in enumerate(backbone):
            if i == 0:
                y.append(module(x))
            else:
                if hasattr(module, '__class__') and module.__class__.__name__ == 'C2f_MMFF':
                    # 多模态C2f模块，传入文本特征
                    y.append(module(y[-1], text_features, training))
                else:
                    # 普通模块
                    if isinstance(module, (nn.Conv2d, nn.Upsample, nn.MaxPool2d)):
                        y.append(module(y[-1]))
                    elif hasattr(module, 'f'):
                        # 处理concat等操作
                        if isinstance(module.f, int):
                            y.append(y[module.f])
                        elif isinstance(module.f, list):
                            y.append(torch.cat([y[j] for j in module.f], 1))
                    else:
                        y.append(module(y[-1]))
        
        # Head部分（保持不变）
        if head is not None:
            head_output = head(y)
        else:
            head_output = y[-1]
        
        return head_output
    
    def train(self, data, imgsz=640, epochs=300, batch=16, workers=32, 
              device='', optimizer='SGD', project='runs/train', name='mmff'):
        """训练方法"""
        # 创建自定义的数据加载器，支持多模态数据
        from ultralytics.yolo.engine.trainer import BaseTrainer
        
        class MultiModalTrainer(BaseTrainer):
            def __init__(self, model, data, **kwargs):
                super().__init__(model, data, **kwargs)
                self.model = model
            
            def preprocess_batch(self, batch):
                """预处理批次数据，提取图像和annotation文本"""
                imgs = batch['img']
                annotations = batch.get('annotations', None)
                
                # 如果没有annotation，从labels生成简单的文本描述
                if annotations is None and 'cls' in batch and 'bboxes' in batch:
                    annotations = self._generate_annotations_from_labels(batch)
                
                return imgs, annotations
            
            def _generate_annotations_from_labels(self, batch):
                """从标签生成annotation文本"""
                batch_size = batch['img'].shape[0]
                annotations = []
                
                for i in range(batch_size):
                    if 'cls' in batch and 'bboxes' in batch:
                        cls_labels = batch['cls'][i] if i < len(batch['cls']) else []
                        bboxes = batch['bboxes'][i] if i < len(batch['bboxes']) else []
                        
                        if len(cls_labels) > 0:
                            # 生成文本描述，例如: "object at [0.1,0.2,0.3,0.4], object at [0.5,0.6,0.7,0.8]"
                            desc_list = []
                            for j, (cls, bbox) in enumerate(zip(cls_labels, bboxes)):
                                if j < 5:  # 最多描述5个目标
                                    desc_list.append(f"object at [{bbox[0]:.2f},{bbox[1]:.2f},{bbox[2]:.2f},{bbox[3]:.2f}]")
                            
                            annotation = ", ".join(desc_list)
                            if len(cls_labels) > 5:
                                annotation += f" and {len(cls_labels)-5} more objects"
                        else:
                            annotation = "no objects in image"
                    else:
                        annotation = "no objects in image"
                    
                    annotations.append(annotation)
                
                return annotations
            
            def forward(self, batch, training=True):
                """前向传播"""
                imgs, annotations = self.preprocess_batch(batch)
                return self.model(imgs, annotations, training=training)
        
        # 使用自定义训练器进行训练
        trainer = MultiModalTrainer(
            model=self,
            data=data,
            imgsz=imgsz,
            epochs=epochs,
            batch=batch,
            workers=workers,
            device=device,
            optimizer=optimizer,
            project=project,
            name=name
        )
        
        trainer.train()

def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--weights',type=str, default='yolov8s.pt', help='loading pretrain weights')
    parser.add_argument('--cfg', type=str, default='ultralytics/models/v8/4-yolov8s-AKConv-SPPFLSKA-Bi-SCDown-FPN.yaml', help='models')
    parser.add_argument('--data', type=str, default='datasets.yaml', help='datasets')
    parser.add_argument('--epochs', type=int, default=300, help='train epoch')
    parser.add_argument('--batch', type=int, default=16, help='total batch size for all GPUs')
    parser.add_argument('--imgsz', type=int, default=640, help='image sizes')
    parser.add_argument('--optimizer', default='SGD', help='use optimizer')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--workers', type=int, default=32, help='maximum number of dataloader workers')
    parser.add_argument('--project', default='runs/train', help='save to project/name')
    parser.add_argument('--name', default='mmff', help='save to project/name')
    parser.add_argument('--text_dropout', type=float, default=0.5, help='text feature dropout probability')
    return parser.parse_args()

if __name__ == '__main__':
    args = main()
    
    # 创建多模态模型
    model = MultiModalYOLO(__train(args.cfg), args.weights)
    
    # 训练模型
    model.train(data=args.data,
                imgsz=args.imgsz,
                epochs=args.epochs,
                batch=args.batch,
                workers=args.workers,
                device=args.device,
                optimizer=args.optimizer,
                project=args.project,
                name=args.name)