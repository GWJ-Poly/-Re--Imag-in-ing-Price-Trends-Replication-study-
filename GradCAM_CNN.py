'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import cv2
from PIL import Image
import os
from tqdm import tqdm

class GradCAM_CNN:
    """
    Grad-CAM可视化类，专门针对您提供的CNN模型结构
    """
    
    def __init__(self, model, target_layer_names=None):
        """
        初始化Grad-CAM
        
        参数:
            model: CNN模型
            target_layer_names: 目标层名称列表，如果为None则使用所有卷积层
        """
        self.model = model
        self.model.eval()
        
        if target_layer_names is None:
            # 默认使用所有三个卷积层
            self.target_layer_names = ['conv_layers.0.0', 'conv_layers.1.0', 'conv_layers.2.0']
        else:
            self.target_layer_names = target_layer_names
            
        self.activations = {}
        self.gradients = {}
        
        # 注册钩子
        self._register_hooks()
    
    def _register_hooks(self):
        """为每个目标层注册前向和反向钩子"""
        def forward_hook(layer_name):
            def hook(module, input, output):
                self.activations[layer_name] = output.detach()
            return hook
        
        def backward_hook(layer_name):
            def hook(module, grad_input, grad_output):
                self.gradients[layer_name] = grad_output[0].detach()
            return hook
        
        # 为每个目标层注册钩子
        for name, module in self.model.named_modules():
            if name in self.target_layer_names:
                module.register_forward_hook(forward_hook(name))
                module.register_backward_hook(backward_hook(name))
    
    def _get_target_layer(self, layer_name):
        """根据名称获取目标层"""
        for name, module in self.model.named_modules():
            if name == layer_name:
                return module
        raise ValueError(f"Layer {layer_name} not found")
    
    def generate_cam(self, input_tensor, target_class=None, layer_name=None):
        """
        生成指定层的Grad-CAM
        
        参数:
            input_tensor: 输入张量 (1, 1, H, W)
            target_class: 目标类别，None表示使用预测类别
            layer_name: 目标层名称
            
        返回:
            cam: Grad-CAM热力图
            prediction: 模型预测概率
        """
        if layer_name is None:
            layer_name = self.target_layer_names[-1]  # 默认使用最后一层
        
        # 前向传播
        output = self.model(input_tensor.unsqueeze(0))
        
        if target_class is None:
            target_class = output.argmax(dim=1).item()
        
        # 清零梯度并反向传播
        self.model.zero_grad()
        one_hot = torch.zeros_like(output)
        one_hot[0, target_class] = 1.0
        output.backward(gradient=one_hot, retain_graph=True)
        
        # 获取激活和梯度
        activations = self.activations[layer_name]
        gradients = self.gradients[layer_name]
        
        # 全局平均池化梯度
        weights = torch.mean(gradients, dim=(2, 3), keepdim=True)
        
        # 计算CAM
        cam = torch.sum(weights * activations, dim=1, keepdim=True)
        cam = F.relu(cam)  # ReLU激活，只保留正响应
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)  # 归一化到[0,1]
        
        return cam.squeeze().cpu().numpy(), F.softmax(output, dim=1).detach().cpu().numpy()
    
    def generate_multilayer_cams(self, input_tensor, target_class=None):
        """为所有目标层生成CAM"""
        cams = {}
        predictions = None
        
        for layer_name in self.target_layer_names:
            cam, pred = self.generate_cam(input_tensor, target_class, layer_name)
            cams[layer_name] = cam
            if predictions is None:
                predictions = pred
                
        return cams, predictions
    
    def overlay_heatmap(self, original_image, cam, alpha=0.5, colormap=cv2.COLORMAP_JET):
        """
        将热力图叠加到原图像上
        """
        # 调整热力图大小匹配原图
        h, w = original_image.shape[:2]
        cam_resized = cv2.resize(cam, (w, h))
        
        # 应用颜色映射
        heatmap = cv2.applyColorMap(np.uint8(255 * cam_resized), colormap)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        
        # 确保原图为uint8
        if original_image.dtype != np.uint8:
            original_image = (original_image * 255).astype(np.uint8)
        
        # 叠加热力图
        overlayed = cv2.addWeighted(original_image, 1 - alpha, heatmap, alpha, 0)
        
        return overlayed

def select_extreme_samples(model, batch_data, batch_labels, num_samples=10):
    """
    从batch中选择最极端的预测样本
    
    参数:
        model: CNN模型
        batch_data: 批量数据 (128, 1, H, W)
        batch_labels: 批量标签 (128,)
        num_samples: 每类选择的样本数
        
    返回:
        up_samples: 最可能上涨的样本列表 [(data, label, prob, index), ...]
        down_samples: 最可能下跌的样本列表 [(data, label, prob, index), ...]
    """
    model.eval()
    
    with torch.no_grad():
        # 获取预测概率
        outputs = model(batch_data)
        probabilities = F.softmax(outputs, dim=1)
        
        # 分离上涨和下跌概率
        up_probs = probabilities[:, 1]  # 类别1的概率（上涨）
        down_probs = probabilities[:, 0]  # 类别0的概率（下跌）
        
        # 选择最极端的样本
        up_indices = torch.topk(up_probs, num_samples).indices
        down_indices = torch.topk(down_probs, num_samples).indices
        
        up_samples = []
        down_samples = []
        
        for idx in up_indices:
            up_samples.append((
                batch_data[idx],
                batch_labels[idx],
                up_probs[idx].item(),
                idx.item()
            ))
        
        for idx in down_indices:
            down_samples.append((
                batch_data[idx],
                batch_labels[idx],
                down_probs[idx].item(),
                idx.item()
            ))
    
    return up_samples, down_samples

def visualize_gradcam_analysis(model, batch_data, batch_labels, output_dir="./gradcam_results"):
    """
    执行完整的Grad-CAM可视化分析（论文第6.3节复现）
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. 选择极端样本
    print("选择极端预测样本...")
    up_samples, down_samples = select_extreme_samples(model, batch_data, batch_labels, 10)
    
    print(f"找到 {len(up_samples)} 个上涨极端样本")
    print(f"找到 {len(down_samples)} 个下跌极端样本")
    
    # 2. 初始化Grad-CAM
    grad_cam = GradCAM_CNN(model)
    
    # 3. 为每个样本生成可视化
    print("生成Grad-CAM可视化...")
    
    # 上涨样本可视化
    visualize_sample_group(
        grad_cam, up_samples, "上涨预测", "up", output_dir
    )
    
    # 下跌样本可视化
    visualize_sample_group(
        grad_cam, down_samples, "下跌预测", "down", output_dir
    )
    
    # 4. 生成对比分析图（类似论文图13）
    generate_comparison_figure(
        grad_cam, up_samples[:5], down_samples[:5], output_dir
    )
    
    print(f"可视化完成！结果保存在: {output_dir}")

def visualize_sample_group(grad_cam, samples, group_name, prefix, output_dir):
    """可视化一组样本"""
    fig, axes = plt.subplots(2, 5, figsize=(20, 8))
    axes = axes.ravel()
    
    for i, (image, label, prob, idx) in enumerate(samples):
        if i >= 10:  # 只显示前10个
            break
            
        # 生成多层CAM
        cams, predictions = grad_cam.generate_multilayer_cams(image)
        
        # 转换为numpy图像
        img_np = image.squeeze().cpu().numpy()
        
        # 归一化到[0,255]
        img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min()) * 255
        img_np = img_np.astype(np.uint8)
        
        # 如果是灰度图，转换为RGB
        if len(img_np.shape) == 2:
            img_rgb = np.stack([img_np] * 3, axis=-1)
        else:
            img_rgb = img_np.transpose(1, 2, 0) if img_np.shape[0] in [1, 3] else img_np
        
        # 使用最后一层的CAM
        final_cam = cams[grad_cam.target_layer_names[-1]]
        
        # 叠加热力图
        overlayed_img = grad_cam.overlay_heatmap(img_rgb, final_cam, alpha=0.5)
        
        # 绘制
        axes[i].imshow(overlayed_img)
        axes[i].set_title(
            f'{group_name} #{i+1}\n'
            f'真值: {"上涨" if label==1 else "下跌"}\n'
            f'预测概率: {prob:.3f}'
        )
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/{prefix}_predictions.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 保存每个样本的详细分析
    save_detailed_analysis(grad_cam, samples, prefix, output_dir)

def save_detailed_analysis(grad_cam, samples, prefix, output_dir):
    """保存每个样本的详细层激活分析"""
    for i, (image, label, prob, idx) in enumerate(samples[:3]):  # 只分析前3个详细样本
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        axes = axes.ravel()
        
        # 生成所有层的CAM
        cams, predictions = grad_cam.generate_multilayer_cams(image)
        
        # 原始图像
        img_np = image.squeeze().cpu().numpy()
        img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min()) * 255
        img_np = img_np.astype(np.uint8)
        
        if len(img_np.shape) == 2:
            img_rgb = np.stack([img_np] * 3, axis=-1)
        else:
            img_rgb = img_np.transpose(1, 2, 0) if img_np.shape[0] in [1, 3] else img_np
        
        # 绘制原始图像
        axes[0].imshow(img_rgb, cmap='gray' if len(img_rgb.shape)==2 else None)
        axes[0].set_title('原始图像')
        axes[0].axis('off')
        
        # 绘制每层的激活图
        for j, layer_name in enumerate(grad_cam.target_layer_names):
            cam = cams[layer_name]
            cam_resized = cv2.resize(cam, (img_rgb.shape[1], img_rgb.shape[0]))
            
            axes[j+1].imshow(cam_resized, cmap='jet')
            axes[j+1].set_title(f'{layer_name} 激活')
            axes[j+1].axis('off')
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/{prefix}_sample_{i+1}_layer_analysis.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()

def generate_comparison_figure(grad_cam, up_samples, down_samples, output_dir):
    """生成上涨vs下跌的对比分析图"""
    fig, axes = plt.subplots(2, 5, figsize=(25, 10))
    
    # 上涨样本
    for i, (image, label, prob, idx) in enumerate(up_samples):
        cams, predictions = grad_cam.generate_multilayer_cams(image)
        final_cam = cams[grad_cam.target_layer_names[-1]]
        
        img_np = image.squeeze().cpu().numpy()
        img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min()) * 255
        img_np = img_np.astype(np.uint8)
        
        if len(img_np.shape) == 2:
            img_rgb = np.stack([img_np] * 3, axis=-1)
        else:
            img_rgb = img_np.transpose(1, 2, 0) if img_np.shape[0] in [1, 3] else img_np
        
        overlayed_img = grad_cam.overlay_heatmap(img_rgb, final_cam, alpha=0.6)
        
        axes[0, i].imshow(overlayed_img)
        axes[0, i].set_title(f'上涨样本 {i+1}\n概率: {prob:.3f}')
        axes[0, i].axis('off')
    
    # 下跌样本
    for i, (image, label, prob, idx) in enumerate(down_samples):
        cams, predictions = grad_cam.generate_multilayer_cams(image)
        final_cam = cams[grad_cam.target_layer_names[-1]]
        
        img_np = image.squeeze().cpu().numpy()
        img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min()) * 255
        img_np = img_np.astype(np.uint8)
        
        if len(img_np.shape) == 2:
            img_rgb = np.stack([img_np] * 3, axis=-1)
        else:
            img_rgb = img_np.transpose(1, 2, 0) if img_np.shape[0] in [1, 3] else img_np
        
        overlayed_img = grad_cam.overlay_heatmap(img_rgb, final_cam, alpha=0.6)
        
        axes[1, i].imshow(overlayed_img)
        axes[1, i].set_title(f'下跌样本 {i+1}\n概率: {prob:.3f}')
        axes[1, i].axis('off')
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/up_vs_down_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()

def analyze_activation_patterns(grad_cam, up_samples, down_samples, output_dir):
    """分析激活模式统计特征（论文第6.3节统计分析）"""
    up_activation_stats = []
    down_activation_stats = []
    
    # 分析上涨样本
    for image, label, prob, idx in up_samples:
        cams, _ = grad_cam.generate_multilayer_cams(image)
        cam = cams[grad_cam.target_layer_names[-1]]  # 使用最后一层
        
        stats = {
            'mean_activation': np.mean(cam),
            'max_activation': np.max(cam),
            'activation_area': np.sum(cam > 0.5) / cam.size,  # 高激活区域比例
            'activation_std': np.std(cam)
        }
        up_activation_stats.append(stats)
    
    # 分析下跌样本
    for image, label, prob, idx in down_samples:
        cams, _ = grad_cam.generate_multilayer_cams(image)
        cam = cams[grad_cam.target_layer_names[-1]]
        
        stats = {
            'mean_activation': np.mean(cam),
            'max_activation': np.max(cam),
            'activation_area': np.sum(cam > 0.5) / cam.size,
            'activation_std': np.std(cam)
        }
        down_activation_stats.append(stats)
    
    # 计算平均值
    up_means = {k: np.mean([d[k] for d in up_activation_stats]) for k in up_activation_stats[0].keys()}
    down_means = {k: np.mean([d[k] for d in down_activation_stats]) for k in down_activation_stats[0].keys()}
    
    # 打印统计结果
    print("\n激活模式统计分析（论文第6.3节）:")
    print("="*50)
    for key in up_means.keys():
        print(f"{key}:")
        print(f"  上涨预测: {up_means[key]:.4f}")
        print(f"  下跌预测: {down_means[key]:.4f}")
        print(f"  差异: {abs(up_means[key] - down_means[key]):.4f}")
        print()
    
    # 绘制比较图
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    metrics = list(up_means.keys())
    
    for i, metric in enumerate(metrics):
        row, col = i // 2, i % 2
        values = [up_means[metric], down_means[metric]]
        axes[row, col].bar(['上涨预测', '下跌预测'], values, color=['green', 'red'], alpha=0.7)
        axes[row, col].set_title(f'{metric} 对比')
        axes[row, col].set_ylabel('数值')
        
        # 添加数值标签
        for j, v in enumerate(values):
            axes[row, col].text(j, v, f'{v:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/activation_patterns_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import pyarrow.feather as feather
import os
from pathlib import Path
from tqdm import tqdm
import config as cf
from datetime import datetime
import torch.nn.functional as F
from cnn_model import Model
def load_pretrained_model(model_path, ws=20, device='cuda'):
    """加载预训练模型"""
    layer_number = cf.BENCHMARK_MODEL_LAYERNUM_DICT[ws]
    filter_size_list, stride_list, dilation_list, max_pooling_list = cf.EMP_CNN_BL_SETTING[ws]
    inplanes = cf.TRUE_DATA_CNN_INPLANES
    
    model_obj = Model(
        ws=ws,
        layer_number=layer_number,
        inplanes=inplanes,
        drop_prob=0.5,
        filter_size_list=filter_size_list,
        stride_list=stride_list,
        dilation_list=dilation_list,
        max_pooling_list=max_pooling_list,
        batch_norm=True,
        xavier=True,
        lrelu=True,
        bn_loc="bn_bf_relu",
        conv_layer_chanls=None,
        regression_label=None,
        ts1d_model=False
    )
    
    model = model_obj.init_model(device=device)
    checkpoint = torch.load(model_path, map_location=device)
    if 'state_dict' in checkpoint:
        model.load_state_dict(checkpoint['state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model.eval()
    print(f"✓ 模型从 {model_path} 加载成功")
    return model
# 使用示例
import os
import numpy as np
import h5py
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import csv
class H5Dataset(Dataset):
    def __init__(self, h5_path):
        self.h5_path = h5_path
        # 只读取 labels 来筛选（一次性读取 labels 是可接受的，因为 labels 很小）
        with h5py.File(self.h5_path, "r") as f:
            labels = f["labels"][:]
        # 我们只保留 label != 2 的样本（2 表示缺失/NaN）
        self.valid_indices = np.where(labels != 2)[0].astype(np.int64)
        self._len = len(self.valid_indices)
        # 不在 __init__ 中打开 h5（避免 multiprocessing 中的文件句柄问题）
        self._h5 = None

    def __len__(self):
        return self._len

    def __getitem__(self, idx):
        # 延迟打开 h5（每个 worker 都会有自己的文件句柄）
        if self._h5 is None:
            self._h5 = h5py.File(self.h5_path, "r")
        real_idx = int(self.valid_indices[idx])
        img = self._h5["images"][real_idx]   # shape (H, W), dtype float32
        label = int(self._h5["labels"][real_idx])  # 0 or 1
        # 转为 Tensor，添加 channel 维度
        img_t = torch.from_numpy(img).unsqueeze(0).float()  # (1,H,W)
        label_t = torch.tensor(label, dtype=torch.long)
        return img_t, label_t

    def close(self):
        if self._h5 is not None:
            try:
                self._h5.close()
            except:
                pass
            self._h5 = None
def main():
    """
    主函数：演示如何使用代码复现论文第6.3节
    """
    model=load_pretrained_model(model_path='/workspace_ssd/wangjiang/monthly_20d/exp/checkpoints_baseline_fix/model_best.pt', ws=20, device='cuda')
    # 假设您已经有以下数据：
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # model: 训练好的CNN模型
    # batch_data: 一个batch的图像数据 (128, 1, H, W)
    # batch_labels: 对应的标签 (128,)
    train_h5 = "/workspace_ssd/wangjiang/monthly_20d/train_1993_1999_ori_20day_c.hdf5"
    val_h5   = "/workspace_ssd/wangjiang/monthly_20d/val_1993_1999_ori_20day_c.hdf5"
    batch_size = 128
    num_workers = 4#16         # DataLoader 的 num_workers（根据你的机器调整）
    train_dataset = H5Dataset(train_h5)
    val_dataset = H5Dataset(val_h5)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True ,
      
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=max(0, num_workers//2),
        pin_memory=True ,
       
    )
    model.to(device)
    model.eval()
    num=0

    for imgs, labels in tqdm(val_loader):
        imgs = imgs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        num+=1
# 示例使用方式：
        visualize_gradcam_analysis(model, imgs, labels)
        if num>10:
            break
    
    print("""
    使用说明：
    1. 确保已安装所需库：torch, matplotlib, numpy, opencv-python, Pillow
    2. 准备模型和batch数据
    3. 调用 visualize_gradcam_analysis(model, batch_data, batch_labels)
    4. 查看生成的图像和分析结果
    """)

if __name__ == "__main__":
    main()
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import cv2
from PIL import Image
import os
from tqdm import tqdm
import h5py
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim

# 定义数据加载类
class H5Dataset(Dataset):
    def __init__(self, h5_path):
        self.h5_path = h5_path
        # 只读取 labels 来筛选（一次性读取 labels 是可接受的，因为 labels 很小）
        with h5py.File(self.h5_path, "r") as f:
            labels = f["labels"][:]
        # 我们只保留 label != 2 的样本（2 表示缺失/NaN）
        self.valid_indices = np.where(labels != 2)[0].astype(np.int64)
        self._len = len(self.valid_indices)
        # 不在 __init__ 中打开 h5（避免 multiprocessing 中的文件句柄问题）
        self._h5 = None

    def __len__(self):
        return self._len

    def __getitem__(self, idx):
        if self._h5 is None:
            self._h5 = h5py.File(self.h5_path, "r")
        real_idx = int(self.valid_indices[idx])
        
        # 读取标签并检查NaN
        label_value = self._h5["labels"][real_idx]
        
        # 检查是否为NaN
        if np.isnan(label_value):
            # 处理方案1: 跳过NaN样本，返回下一个有效样本
            # return self.__getitem__((idx + 1) % len(self))
            # 处理方案2: 抛出明确异常
            raise ValueError(f"在索引 {real_idx} 处发现NaN标签")
        
        # 确保标签是有效整数
        label = int(label_value)
        
        img = self._h5["images"][real_idx]
        img_t = torch.from_numpy(img).unsqueeze(0).float()
        label_t = torch.tensor(label, dtype=torch.long)
        
        return img_t, label_t
    def close(self):
        if self._h5 is not None:
            try:
                self._h5.close()
            except:
                pass
            self._h5 = None

# 定义CNN模型（与您提供的结构匹配）
class CNNModel(nn.Module):
    def __init__(self, input_channels=1, num_classes=2):
        super(CNNModel, self).__init__()
        
        self.conv_layers = nn.Sequential(
            # 第一层
            nn.Sequential(
                nn.Conv2d(input_channels, 64, kernel_size=(5, 3), stride=(3, 1), padding=(2, 1), dilation=(2, 1)),
                nn.BatchNorm2d(64),
                nn.LeakyReLU(negative_slope=0.01),
                nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1))
            ),
            # 第二层
            nn.Sequential(
                nn.Conv2d(64, 128, kernel_size=(5, 3), stride=(1, 1), padding=(2, 1)),
                nn.BatchNorm2d(128),
                nn.LeakyReLU(negative_slope=0.01),
                nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1))
            ),
            # 第三层
            nn.Sequential(
                nn.Conv2d(128, 256, kernel_size=(5, 3), stride=(1, 1), padding=(2, 1)),
                nn.BatchNorm2d(256),
                nn.LeakyReLU(negative_slope=0.01),
                nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1))
            ),
            # 展平层
            self.Flatten(),
            nn.Dropout(p=0.5)
        )
        
        # 计算全连接层输入尺寸
        self.fc = nn.Linear(46080, num_classes)
    
    class Flatten(nn.Module):
        def forward(self, x):
            return x.view(x.size(0), -1)
    
    def forward(self, x):
        x = self.conv_layers(x)
        x = self.fc(x)
        return x

# Grad-CAM类
class GradCAM_CNN:
    def __init__(self, model, target_layer_names=None):
        self.model = model
        self.model.eval()
        
        if target_layer_names is None:
            # 使用三个卷积层
            self.target_layer_names = ['conv_layers.0.0', 'conv_layers.1.0', 'conv_layers.2.0']
        else:
            self.target_layer_names = target_layer_names
            
        self.activations = {}
        self.gradients = {}
        
        self._register_hooks()
    
    def _register_hooks(self):
        def forward_hook(layer_name):
            def hook(module, input, output):
                self.activations[layer_name] = output.detach()
            return hook
        
        def backward_hook(layer_name):
            def hook(module, grad_input, grad_output):
                self.gradients[layer_name] = grad_output[0].detach()
            return hook
        
        for name, module in self.model.named_modules():
            if name in self.target_layer_names:
                module.register_forward_hook(forward_hook(name))
                module.register_backward_hook(backward_hook(name))
    
    def generate_cam(self, input_tensor, target_class=None, layer_name=None):
        if layer_name is None:
            layer_name = self.target_layer_names[-1]
        
        # 前向传播
        output = self.model(input_tensor.unsqueeze(0))
        
        if target_class is None:
            target_class = output.argmax(dim=1).item()
        
        # 反向传播
        self.model.zero_grad()
        one_hot = torch.zeros_like(output)
        one_hot[0, target_class] = 1.0
        output.backward(gradient=one_hot, retain_graph=True)
        
        # 获取激活和梯度
        activations = self.activations[layer_name]
        gradients = self.gradients[layer_name]
        
        # 全局平均池化梯度
        weights = torch.mean(gradients, dim=(2, 3), keepdim=True)
        
        # 计算CAM
        cam = torch.sum(weights * activations, dim=1, keepdim=True)
        cam = F.relu(cam)
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)
        
        return cam.squeeze().cpu().numpy(), F.softmax(output, dim=1).detach().cpu().numpy()
    
    def generate_multilayer_cams(self, input_tensor, target_class=None):
        cams = {}
        predictions = None
        
        for layer_name in self.target_layer_names:
            cam, pred = self.generate_cam(input_tensor, target_class, layer_name)
            cams[layer_name] = cam
            if predictions is None:
                predictions = pred
                
        return cams, predictions

# 选择极端样本函数
def select_correct_extreme_samples(model, batch_data, batch_labels, num_samples=10):
    """
    修正版本：选择预测正确且置信度最高的样本
    - 上涨正确样本：真实标签=1且预测为1（上涨）概率最高的样本
    - 下跌正确样本：真实标签=0且预测为0（下跌）概率最高的样本
    """
    model.eval()
    
    with torch.no_grad():
        # 获取模型预测
        outputs = model(batch_data)
        probabilities = F.softmax(outputs, dim=1)
        #print(probabilities.shape)
        predictions = outputs.argmax(dim=1)  # 获取预测类别
        
        # 找出预测正确的样本
        correct_mask = (predictions == batch_labels)
        correct_indices = torch.where(correct_mask)[0]
        
        if len(correct_indices) == 0:
            print("本批次中没有预测正确的样本")
            return [], []
        
        # 分离正确预测的上涨和下跌样本
        up_correct_mask = (batch_labels[correct_indices] == 1)  # 真实标签为1且预测正确
        down_correct_mask = (batch_labels[correct_indices] == 0)  # 真实标签为0且预测正确
        
        up_correct_indices = correct_indices[up_correct_mask]
        down_correct_indices = correct_indices[down_correct_mask]
        
        print(f"正确样本统计: 上涨正确 {len(up_correct_indices)}个, 下跌正确 {len(down_correct_indices)}个")
        
        # 选择置信度最高的正确样本
        up_samples = []
        down_samples = []
        
        # 处理上涨正确样本
        if len(up_correct_indices) > 0:
            up_probs = probabilities[up_correct_indices, 1]  # 上涨样本的上涨概率
            
            # 修正：正确的torch.topk使用方式
            k = min(num_samples, len(up_probs))
            if k > 0:  # 确保k大于0
                _, topk_indices = torch.topk(up_probs, k)  # 返回(values, indices)元组
                up_top_indices = up_correct_indices[topk_indices]
                
                for idx in up_top_indices:
                    true_label = batch_labels[idx].item()
                    pred_prob = probabilities[idx, 1].item()  # 预测为上涨的概率
                    up_samples.append((
                        batch_data[idx],
                        true_label,
                        pred_prob,
                        idx.item()
                    ))
        
        # 处理下跌正确样本
        if len(down_correct_indices) > 0:
            down_probs = probabilities[down_correct_indices, 0]  # 下跌样本的下跌概率
            
            # 修正：正确的torch.topk使用方式
            k = min(num_samples, len(down_probs))
            if k > 0:  # 确保k大于0
                _, topk_indices = torch.topk(down_probs, k)  # 返回(values, indices)元组
                down_top_indices = down_correct_indices[topk_indices]
                
                for idx in down_top_indices:
                    true_label = batch_labels[idx].item()
                    pred_prob = probabilities[idx, 0].item()  # 预测为下跌的概率
                    down_samples.append((
                        batch_data[idx],
                        true_label,
                        pred_prob,
                        idx.item()
                    ))
    
    print(f"样本选择结果: {len(up_samples)}个上涨正确样本, {len(down_samples)}个下跌正确样本")
    
    # 验证选择逻辑
    if len(up_samples) > 0:
        min_prob = min([s[2] for s in up_samples])
        max_prob = max([s[2] for s in up_samples])
        print(f"上涨正确样本预测概率范围: {min_prob:.3f} - {max_prob:.3f}")
    
    if len(down_samples) > 0:
        min_prob = min([s[2] for s in down_samples])
        max_prob = max([s[2] for s in down_samples])
        print(f"下跌正确样本预测概率范围: {min_prob:.3f} - {max_prob:.3f}")
    
    return up_samples, down_samples
'''
def create_figure13_style_plot(grad_cam, up_samples, down_samples, output_path, batch_idx):
    """
    修正版本：准确反映样本选择标准
    上半部分：预测为上涨概率最高的10个样本
    下半部分：预测为下跌概率最高的10个样本
    """
    # 创建图形（8行10列：4行上涨预测样本 + 4行下跌预测样本）
    fig, axes = plt.subplots(8, 10, figsize=(25, 20))
    
    # 设置准确的标题
    fig.suptitle(f'Grad-CAM Analysis: Highest Prediction Confidence Samples (Batch {batch_idx})', 
                fontsize=18, fontweight='bold', y=0.95)
    
    # 处理预测为上涨概率最高的样本（上半部分）
    for col, (image, true_label, pred_prob, idx) in enumerate(up_samples[:10]):
        # 生成多层CAM
        cams, predictions = grad_cam.generate_multilayer_cams(image)
        
        # 转换为numpy图像
        img_np = image.squeeze().cpu().numpy()
        img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min()) * 255
        img_np = img_np.astype(np.uint8)
        
        # 如果是灰度图，转换为RGB
        if len(img_np.shape) == 2:
            img_rgb = np.stack([img_np] * 3, axis=-1)
        else:
            img_rgb = img_np
        
        # 第一行：原始图像（上涨预测样本）
        ax = axes[0, col]
        ax.imshow(img_rgb, cmap='gray')
        # 修正标题：明确显示这是预测为上涨概率最高的样本
        ax.set_title(f'Top Up-Pred #{col+1}\nTrue: {"Up" if true_label==1 else "Down"}\nPred Up: {pred_prob:.3f}', 
                    fontsize=9)
        ax.axis('off')
        
        # 第二行：第一层激活（上涨预测样本）
        ax = axes[1, col]
        cam1 = cams['conv_layers.0.0']
        cam1_resized = cv2.resize(cam1, (img_rgb.shape[1], img_rgb.shape[0]))
        ax.imshow(cam1_resized, cmap='jet', alpha=0.7)
        ax.imshow(img_rgb, cmap='gray', alpha=0.3)
        ax.set_title('Layer 1 Activation', fontsize=9)
        ax.axis('off')
        
        # 第三行：第二层激活（上涨预测样本）
        ax = axes[2, col]
        cam2 = cams['conv_layers.1.0']
        cam2_resized = cv2.resize(cam2, (img_rgb.shape[1], img_rgb.shape[0]))
        ax.imshow(cam2_resized, cmap='jet', alpha=0.7)
        ax.imshow(img_rgb, cmap='gray', alpha=0.3)
        ax.set_title('Layer 2 Activation', fontsize=9)
        ax.axis('off')
        
        # 第四行：第三层激活（上涨预测样本）
        ax = axes[3, col]
        cam3 = cams['conv_layers.2.0']
        cam3_resized = cv2.resize(cam3, (img_rgb.shape[1], img_rgb.shape[0]))
        ax.imshow(cam3_resized, cmap='jet', alpha=0.7)
        ax.imshow(img_rgb, cmap='gray', alpha=0.3)
        ax.set_title('Layer 3 Activation', fontsize=9)
        ax.axis('off')
    
    # 处理预测为下跌概率最高的样本（下半部分）
    for col, (image, true_label, pred_prob, idx) in enumerate(down_samples[:10]):
        # 生成多层CAM
        cams, predictions = grad_cam.generate_multilayer_cams(image)
        
        # 转换为numpy图像
        img_np = image.squeeze().cpu().numpy()
        img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min()) * 255
        img_np = img_np.astype(np.uint8)
        
        # 如果是灰度图，转换为RGB
        if len(img_np.shape) == 2:
            img_rgb = np.stack([img_np] * 3, axis=-1)
        else:
            img_rgb = img_np
        
        # 第五行：原始图像（下跌预测样本）
        ax = axes[4, col]
        ax.imshow(img_rgb, cmap='gray')
        # 修正标题：明确显示这是预测为下跌概率最高的样本
        ax.set_title(f'Top Down-Pred #{col+1}\nTrue: {"Up" if true_label==1 else "Down"}\nPred Down: {pred_prob:.3f}', 
                    fontsize=9)
        ax.axis('off')
        
        # 第六行：第一层激活（下跌预测样本）
        ax = axes[5, col]
        cam1 = cams['conv_layers.0.0']
        cam1_resized = cv2.resize(cam1, (img_rgb.shape[1], img_rgb.shape[0]))
        ax.imshow(cam1_resized, cmap='jet', alpha=0.7)
        ax.imshow(img_rgb, cmap='gray', alpha=0.3)
        ax.set_title('Layer 1 Activation', fontsize=9)
        ax.axis('off')
        
        # 第七行：第二层激活（下跌预测样本）
        ax = axes[6, col]
        cam2 = cams['conv_layers.1.0']
        cam2_resized = cv2.resize(cam2, (img_rgb.shape[1], img_rgb.shape[0]))
        ax.imshow(cam2_resized, cmap='jet', alpha=0.7)
        ax.imshow(img_rgb, cmap='gray', alpha=0.3)
        ax.set_title('Layer 2 Activation', fontsize=9)
        ax.axis('off')
        
        # 第八行：第三层激活（下跌预测样本）
        ax = axes[7, col]
        cam3 = cams['conv_layers.2.0']
        cam3_resized = cv2.resize(cam3, (img_rgb.shape[1], img_rgb.shape[0]))
        ax.imshow(cam3_resized, cmap='jet', alpha=0.7)
        ax.imshow(img_rgb, cmap='gray', alpha=0.3)
        ax.set_title('Layer 3 Activation', fontsize=9)
        ax.axis('off')
    
    # 修正分区标签，准确描述选择标准
    plt.figtext(0.02, 0.45, '(a) Top 10 Samples with Highest "Up" Prediction Probability', 
               fontsize=14, fontweight='bold', ha='left')
    plt.figtext(0.02, 0.05, '(b) Top 10 Samples with Highest "Down" Prediction Probability', 
               fontsize=14, fontweight='bold', ha='left')
    
    # 修正注释，明确选择标准
    
    plt.figtext(0.5, 0.02, 'Note: Samples selected based solely on model prediction probability (not ground truth accuracy). '
                          'Top row: 10 samples with highest probability of being predicted as "Up". '
                          'Bottom row: 10 samples with highest probability of being predicted as "Down". '
                          'True label shown for reference only.',
               fontsize=10, ha='center', style='italic')
    
    plt.tight_layout(rect=[0.03, 0.03, 0.97, 0.95])
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"图13风格可视化已保存: {output_path}")
'''
def create_compact_comparison_plot(grad_cam, up_samples, down_samples, output_path, batch_idx):
    import matplotlib.pyplot as plt
    import numpy as np
    import cv2

    # 画布：显式关闭 constrained_layout（若不关会覆盖 subplots_adjust）
    fig = plt.figure(figsize=(24, 20), constrained_layout=False)

    # 使用 add_gridspec 并显式给出小的 wspace/hspace（正值）
    gs = fig.add_gridspec(nrows=10, ncols=10,
                          height_ratios=[1, 1, 1, 1, 0.8, 1, 1, 1, 1, 0.2],
                          wspace=0.1,hspace=0.2)

    plt.rcParams.update({'font.size': 16})

    def prepare_image(image):
        img_np = image.squeeze().cpu().numpy()
        denom = (img_np.max() - img_np.min())
        if denom == 0:
            norm = np.zeros_like(img_np)
        else:
            norm = (img_np - img_np.min()) / denom * 255
        img_np = norm.astype(np.uint8)
        if img_np.ndim == 2:
            img_rgb = np.stack([img_np] * 3, axis=-1)
        else:
            img_rgb = img_np
        return img_rgb

    # 上半部分 (rows 0-3)
    for col, (image, true_label, pred_prob, idx) in enumerate(up_samples[:10]):
        cams, _ = grad_cam.generate_multilayer_cams(image)
        img_rgb = prepare_image(image)

        ax = fig.add_subplot(gs[0, col])
        ax.imshow(img_rgb, cmap='gray', aspect='auto')    # aspect='auto' 更紧凑
        ax.set_title(f'Up#{col+1}\nTrue:{"Up" if true_label==1 else "Down"}\nP:{pred_prob:.3f}',
                     fontsize=16, pad=2)
        ax.axis('off')

        cam1 = cams.get('conv_layers.0.0', None) if isinstance(cams, dict) else (cams[0] if len(cams)>0 else None)
        ax = fig.add_subplot(gs[1, col])
        if cam1 is None:
            ax.imshow(img_rgb, aspect='auto')
        else:
            cam1_resized = cv2.resize(cam1, (img_rgb.shape[1], img_rgb.shape[0]))
            ax.imshow(cam1_resized, cmap='jet', alpha=0.7, aspect='auto')
            ax.imshow(img_rgb, cmap='gray', alpha=0.3, aspect='auto')
        ax.set_title('L1', fontsize=16, pad=1)
        ax.axis('off')

        cam2 = cams.get('conv_layers.1.0', None) if isinstance(cams, dict) else (cams[1] if len(cams)>1 else None)
        ax = fig.add_subplot(gs[2, col])
        if cam2 is None:
            ax.imshow(img_rgb, aspect='auto')
        else:
            cam2_resized = cv2.resize(cam2, (img_rgb.shape[1], img_rgb.shape[0]))
            ax.imshow(cam2_resized, cmap='jet', alpha=0.7, aspect='auto')
            ax.imshow(img_rgb, cmap='gray', alpha=0.3, aspect='auto')
        ax.set_title('L2', fontsize=16, pad=1)
        ax.axis('off')

        cam3 = cams.get('conv_layers.2.0', None) if isinstance(cams, dict) else (cams[2] if len(cams)>2 else None)
        ax = fig.add_subplot(gs[3, col])
        if cam3 is None:
            ax.imshow(img_rgb, aspect='auto')
        else:
            cam3_resized = cv2.resize(cam3, (img_rgb.shape[1], img_rgb.shape[0]))
            ax.imshow(cam3_resized, cmap='jet', alpha=0.7, aspect='auto')
            ax.imshow(img_rgb, cmap='gray', alpha=0.3, aspect='auto')
        ax.set_title('L3', fontsize=16, pad=1)
        ax.axis('off')

    # caption (a)
    fig.text(0.5, 0.52, '(a) 10 Samples with "Up" Prediction Probability',
             ha='center', va='center', fontsize=20, fontweight='bold')

    # 下半部分 (rows 5-8)
    for col, (image, true_label, pred_prob, idx) in enumerate(down_samples[:10]):
        cams, _ = grad_cam.generate_multilayer_cams(image)
        img_rgb = prepare_image(image)

        ax = fig.add_subplot(gs[5, col])
        ax.imshow(img_rgb, cmap='gray', aspect='auto')
        ax.set_title(f'Down#{col+1}\nTrue:{"Up" if true_label==1 else "Down"}\nP:{pred_prob:.3f}',
                     fontsize=16, pad=2)
        ax.axis('off')

        cam1 = cams.get('conv_layers.0.0', None) if isinstance(cams, dict) else (cams[0] if len(cams)>0 else None)
        ax = fig.add_subplot(gs[6, col])
        if cam1 is None:
            ax.imshow(img_rgb, aspect='auto')
        else:
            cam1_resized = cv2.resize(cam1, (img_rgb.shape[1], img_rgb.shape[0]))
            ax.imshow(cam1_resized, cmap='jet', alpha=0.7, aspect='auto')
            ax.imshow(img_rgb, cmap='gray', alpha=0.3, aspect='auto')
        ax.set_title('L1', fontsize=16, pad=1)
        ax.axis('off')

        cam2 = cams.get('conv_layers.1.0', None) if isinstance(cams, dict) else (cams[1] if len(cams)>1 else None)
        ax = fig.add_subplot(gs[7, col])
        if cam2 is None:
            ax.imshow(img_rgb, aspect='auto')
        else:
            cam2_resized = cv2.resize(cam2, (img_rgb.shape[1], img_rgb.shape[0]))
            ax.imshow(cam2_resized, cmap='jet', alpha=0.7, aspect='auto')
            ax.imshow(img_rgb, cmap='gray', alpha=0.3, aspect='auto')
        ax.set_title('L2', fontsize=16, pad=1)
        ax.axis('off')

        cam3 = cams.get('conv_layers.2.0', None) if isinstance(cams, dict) else (cams[2] if len(cams)>2 else None)
        ax = fig.add_subplot(gs[8, col])
        if cam3 is None:
            ax.imshow(img_rgb, aspect='auto')
        else:
            cam3_resized = cv2.resize(cam3, (img_rgb.shape[1], img_rgb.shape[0]))
            ax.imshow(cam3_resized, cmap='jet', alpha=0.7, aspect='auto')
            ax.imshow(img_rgb, cmap='gray', alpha=0.3, aspect='auto')
        ax.set_title('L3', fontsize=16, pad=1)
        ax.axis('off')

    # caption (b)
    fig.text(0.5, 0.01, '(b) 10 Samples with "Down" Prediction Probability',
             ha='center', va='center', fontsize=20, fontweight='bold')

    # 最终微调：使用小的正 wspace/hspace，不要用负值
    plt.subplots_adjust(left=0.02, right=0.98, top=0.98, bottom=0.02, hspace=0.1, wspace=0.2)

    plt.savefig(output_path, dpi=300, bbox_inches='tight', pad_inches=0.02)
    plt.close(fig)
    print(f"紧凑布局可视化已保存: {output_path}")


# 更简洁的版本，使用subplots_adjust进行精细控制
def create_compact_comparison_plot_v2(grad_cam, up_samples, down_samples, output_path, batch_idx):
    """
    版本2：使用subplots_adjust进行更精细的布局控制
    """
    # 创建图形
    fig, axes = plt.subplots(8, 10, figsize=(22, 16))
    
    # 设置全局字体
    plt.rcParams.update({'font.size': 8})
    
    # 处理上涨样本（上半部分）
    for col, (image, true_label, pred_prob, idx) in enumerate(up_samples[:10]):
        # 生成多层CAM和图像处理（与之前相同）
        cams, predictions = grad_cam.generate_multilayer_cams(image)
        img_np = image.squeeze().cpu().numpy()
        img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min()) * 255
        img_np = img_np.astype(np.uint8)
        
        if len(img_np.shape) == 2:
            img_rgb = np.stack([img_np] * 3, axis=-1)
        else:
            img_rgb = img_np
        
        # 第一行：原始图像
        ax = axes[0, col]
        ax.imshow(img_rgb, cmap='gray')
        ax.set_title(f'Up#{col+1}\nTrue:{"Up" if true_label==1 else "Down"}\nProb:{pred_prob:.3f}', 
                    fontsize=7, pad=2)
        ax.axis('off')
        
        # 第二至四行：各层激活图
        for row, layer_name in enumerate(['conv_layers.0.0', 'conv_layers.1.0', 'conv_layers.2.0'], 1):
            ax = axes[row, col]
            cam = cams[layer_name]
            cam_resized = cv2.resize(cam, (img_rgb.shape[1], img_rgb.shape[0]))
            ax.imshow(cam_resized, cmap='jet', alpha=0.7)
            ax.imshow(img_rgb, cmap='gray', alpha=0.3)
            ax.set_title(f'L{row}', fontsize=6, pad=1)
            ax.axis('off')
    
    # 处理下跌样本（下半部分）
    for col, (image, true_label, pred_prob, idx) in enumerate(down_samples[:10]):
        # 生成多层CAM和图像处理（与之前相同）
        cams, predictions = grad_cam.generate_multilayer_cams(image)
        img_np = image.squeeze().cpu().numpy()
        img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min()) * 255
        img_np = img_np.astype(np.uint8)
        
        if len(img_np.shape) == 2:
            img_rgb = np.stack([img_np] * 3, axis=-1)
        else:
            img_rgb = img_np
        
        # 第五行：原始图像
        ax = axes[4, col]
        ax.imshow(img_rgb, cmap='gray')
        ax.set_title(f'Down#{col+1}\nTrue:{"Up" if true_label==1 else "Down"}\nProb:{pred_prob:.3f}', 
                    fontsize=7, pad=2)
        ax.axis('off')
        
        # 第六至八行：各层激活图
        for row, layer_name in enumerate(['conv_layers.0.0', 'conv_layers.1.0', 'conv_layers.2.0'], 5):
            ax = axes[row, col]
            cam = cams[layer_name]
            cam_resized = cv2.resize(cam, (img_rgb.shape[1], img_rgb.shape[0]))
            ax.imshow(cam_resized, cmap='jet', alpha=0.7)
            ax.imshow(img_rgb, cmap='gray', alpha=0.3)
            ax.set_title(f'L{row-4}', fontsize=6, pad=1)
            ax.axis('off')
    
    # 添加组图标注 - 使用figtext进行精确定位
    fig.text(0.5, 0.52, '(a) Top 10 Samples with Highest "Up" Prediction Probability', 
            fontsize=12, fontweight='bold', ha='center', va='center')
    fig.text(0.5, 0.06, '(b) Top 10 Samples with Highest "Down" Prediction Probability', 
            fontsize=12, fontweight='bold', ha='center', va='center')
    
    # 精细调整子图间距
    plt.subplots_adjust(
        top=0.94,      # 上边距
        bottom=0.08,   # 下边距，为标注留出空间
        left=0.04,     # 左边距
        right=0.98,    # 右边距
        hspace=0.15,   # 行间距（组图内部）
        wspace=0.15,   # 列间距
    )
    
    # 添加整体说明
    fig.text(0.5, 0.02, 'Note: Brighter heatmap regions indicate higher activation. L1, L2, L3 represent different CNN layers.',
            fontsize=9, ha='center', style='italic')
    
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"紧凑布局可视化已保存: {output_path}")

# 主可视化函数
def visualize_gradcam_analysis_corrected(model, batch_data, batch_labels, output_dir, batch_idx):
    """
    修正版本的主可视化函数
    使用正确的样本选择逻辑
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. 使用修正的样本选择函数
    up_samples, down_samples = select_correct_extreme_samples(model, batch_data, batch_labels, 10)
    
    print(f"批次 {batch_idx}:")
    print(f"  找到 {len(up_samples)} 个预测为上涨概率最高的样本")
    print(f"  找到 {len(down_samples)} 个预测为下跌概率最高的样本")
    
    # 验证选择逻辑
    if len(up_samples) > 0 and len(down_samples) > 0:
        avg_up_prob = np.mean([s[2] for s in up_samples])
        avg_down_prob = np.mean([s[2] for s in down_samples])
        print(f"  平均预测概率 - 上涨: {avg_up_prob:.3f}, 下跌: {avg_down_prob:.3f}")
    
    # 2. 初始化Grad-CAM
    grad_cam = GradCAM_CNN(model)
    
    # 3. 创建修正的可视化
    output_path = os.path.join(output_dir, f'corrected_figure13_batch_{batch_idx}.pdf')
    create_compact_comparison_plot(grad_cam, up_samples, down_samples, output_path, batch_idx)
    
    # 4. 保存详细的样本信息用于验证
    save_sample_info(up_samples, down_samples, output_dir, batch_idx)
    
    return len(up_samples), len(down_samples)

def save_sample_info(up_samples, down_samples, output_dir, batch_idx):
    """保存样本详细信息用于验证选择逻辑"""
    info_file = os.path.join(output_dir, f'sample_info_batch_{batch_idx}.txt')
    
    with open(info_file, 'w') as f:
        f.write("样本选择详细信息验证\n")
        f.write("=" * 50 + "\n\n")
        
        f.write("上涨预测样本（预测为1概率最高的10个）:\n")
        f.write("-" * 30 + "\n")
        for i, (img, true_label, pred_prob, idx) in enumerate(up_samples):
            f.write(f"样本 {i+1}: 索引={idx}, 真实标签={true_label}, 预测上涨概率={pred_prob:.4f}\n")
        
        f.write("\n下跌预测样本（预测为0概率最高的10个）:\n")
        f.write("-" * 30 + "\n")
        for i, (img, true_label, pred_prob, idx) in enumerate(down_samples):
            f.write(f"样本 {i+1}: 索引={idx}, 真实标签={true_label}, 预测下跌概率={pred_prob:.4f}\n")
    
    print(f"样本详细信息已保存: {info_file}")

def create_detailed_analysis(grad_cam, up_samples, down_samples, output_dir, batch_idx):
    """
    为每个样本创建详细分析图
    """
    detailed_dir = os.path.join(output_dir, f'batch_{batch_idx}_detailed')
    os.makedirs(detailed_dir, exist_ok=True)
    
    # 分析上涨样本
    for i, (image, label, prob, idx) in enumerate(up_samples[:3]):  # 只分析前3个
        create_sample_analysis(grad_cam, image, label, prob, idx, 'up', i+1, detailed_dir)
    
    # 分析下跌样本
    for i, (image, label, prob, idx) in enumerate(down_samples[:3]):  # 只分析前3个
        create_sample_analysis(grad_cam, image, label, prob, idx, 'down', i+1, detailed_dir)

def create_sample_analysis(grad_cam, image, label, prob, idx, direction, sample_num, output_dir):
    """
    为单个样本创建详细分析
    """
    # 生成多层CAM
    cams, predictions = grad_cam.generate_multilayer_cams(image)
    
    # 转换为numpy图像
    img_np = image.squeeze().cpu().numpy()
    img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min()) * 255
    img_np = img_np.astype(np.uint8)
    
    # 创建分析图
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # 原始图像
    axes[0, 0].imshow(img_np, cmap='gray')
    axes[0, 0].set_title('Original Image')
    axes[0, 0].axis('off')
    
    # 各层激活
    layers = ['conv_layers.0.0', 'conv_layers.1.0', 'conv_layers.2.0']
    titles = ['Layer 1 Activation', 'Layer 2 Activation', 'Layer 3 Activation']
    
    for i, (layer, title) in enumerate(zip(layers, titles)):
        row, col = (i+1) // 2, (i+1) % 2
        cam = cams[layer]
        cam_resized = cv2.resize(cam, (img_np.shape[1], img_np.shape[0]))
        
        axes[row, col].imshow(img_np, cmap='gray', alpha=0.3)
        axes[row, col].imshow(cam_resized, cmap='jet', alpha=0.7)
        axes[row, col].set_title(title)
        axes[row, col].axis('off')
    
    plt.suptitle(f'{direction.capitalize()} Sample {sample_num}\nTrue: {"Up" if label==1 else "Down"}, Pred: {prob:.3f}', 
                fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(os.path.join(output_dir, f'{direction}_sample_{sample_num}_detailed.png'), 
               dpi=300, bbox_inches='tight')
    plt.close()

# 加载预训练模型
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import pyarrow.feather as feather
import os
from pathlib import Path
from tqdm import tqdm
import config as cf
from datetime import datetime
import torch.nn.functional as F
# 图像尺寸配置
IMAGE_WIDTH = {5: 15, 20: 60, 60: 180}
IMAGE_HEIGHT = {5: 32, 20: 64, 60: 96}

# 假设您有自定义的模型文件
from cnn_model import Model

def load_pretrained_model(model_path, ws=20, device='cuda'):
    """加载预训练模型"""
    layer_number = cf.BENCHMARK_MODEL_LAYERNUM_DICT[ws]
    filter_size_list, stride_list, dilation_list, max_pooling_list = cf.EMP_CNN_BL_SETTING[ws]
    inplanes = cf.TRUE_DATA_CNN_INPLANES
    
    model_obj = Model(
        ws=ws,
        layer_number=layer_number,
        inplanes=inplanes,
        drop_prob=0.5,
        filter_size_list=filter_size_list,
        stride_list=stride_list,
        dilation_list=dilation_list,
        max_pooling_list=max_pooling_list,
        batch_norm=True,
        xavier=True,
        lrelu=True,
        bn_loc="bn_bf_relu",
        conv_layer_chanls=None,
        regression_label=None,
        ts1d_model=False
    )
    
    model = model_obj.init_model(device=device)
    checkpoint = torch.load(model_path, map_location=device)
    if 'state_dict' in checkpoint:
        model.load_state_dict(checkpoint['state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model.eval()
    print(f"✓ 模型从 {model_path} 加载成功")
    return model

# 主函数
def main():
    """
    主函数：演示如何使用代码复现论文第6.3节
    """
    # 加载预训练模型
    model = load_pretrained_model(
        model_path='/workspace_ssd/wangjiang/monthly_20d/exp/checkpoints_baseline_fix/model_best.pt', 
        ws=20, 
        device='cuda'
    )
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    
    # 数据路径
    train_h5 = "/workspace_hdd/wangjiang/monthly_20d/train_1993_1999_ori.hdf5"
    val_h5   = "/workspace_hdd/wangjiang/monthly_20d/val_1993_1999_ori.hdf5"
    
    batch_size = 2000
    num_workers = 4
    
    # 创建数据加载器
    val_dataset = H5Dataset(val_h5)
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    
    # 输出目录
    output_dir = "./gradcam_results_6.3_fix"
    os.makedirs(output_dir, exist_ok=True)
    
    # 处理10个batch
    num_batches = 0
    total_up = 0
    total_down = 0
    
    num_batches = 0
    
    for batch_idx, (imgs, labels) in enumerate(tqdm(val_loader, desc="处理批次")):
        if num_batches >= 10:
            break
            
        imgs = imgs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        
        # 使用修正的可视化函数
        n_up, n_down = visualize_gradcam_analysis_corrected(
            model, imgs, labels, output_dir, batch_idx
        )
        
        num_batches += 1
    
    print(f"\n修正版本处理完成！共处理 {num_batches} 个批次")
    
    print(f"\n处理完成！共处理 {num_batches} 个批次")
    print(f"总上涨样本: {total_up}, 总下跌样本: {total_down}")
    print(f"结果保存在: {output_dir}")

if __name__ == "__main__":
    main()
