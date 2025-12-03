'''
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import pyarrow.feather as feather
from pathlib import Path
import os
from tqdm import tqdm
import config as cf

# 图像尺寸配置
IMAGE_WIDTH = {5: 15, 20: 60, 60: 180}
IMAGE_HEIGHT = {5: 32, 20: 64, 60: 96}

# 假设您有自定义的模型文件
from cnn_model import Model

def load_pretrained_model(model_path, ws=20, device='cuda'):
    """
    加载预训练好的CNN模型
    """
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
    print(f"模型从 {model_path} 加载成功")
    print(f"模型参数量: {sum(p.numel() for p in model.parameters()):,}")
    
    return model

def load_test_images_optimized(dat_file_path, ws=20):
    """
    优化版：加载图像数据并确保数据质量
    """
    height = IMAGE_HEIGHT[ws]
    width = IMAGE_WIDTH[ws]
    
    # 使用memmap内存映射读取
    images_memmap = np.memmap(dat_file_path, mode='r')
    
    # 直接重塑，与您的方法一致
    total_elements = len(images_memmap)
    samples = total_elements // (height * width)
    
    images_reshaped = images_memmap[:samples * height * width].reshape(samples, height, width)
    
    # 数据质量检查
    nan_count = np.isnan(images_reshaped).sum()
    inf_count = np.isinf(images_reshaped).sum()
    
    print(f"加载图像数据: {images_reshaped.shape}")
    print(f"数据范围: [{images_reshaped.min():.6f}, {images_reshaped.max():.6f}]")
    print(f"NaN值数量: {nan_count}, Inf值数量: {inf_count}")
    
    # 处理NaN值（关键修复）
    if nan_count > 0 or inf_count > 0:
        print("处理无效值...")
        images_cleaned = np.nan_to_num(images_reshaped, nan=0.0, posinf=0.0, neginf=0.0)
    else:
        images_cleaned = images_reshaped.copy()
    
    # 转换为tensor
    images_tensor = torch.from_numpy(images_cleaned).unsqueeze(1).float()
    
    # 清理
    del images_memmap
    
    return images_tensor, samples

def load_labels_complete(feather_path):
    """
    完整加载标签数据，保留所有原始列
    """
    df = pd.read_feather(feather_path)
    
    print(f"加载标签数据: {df.shape}")
    print(f"时间范围: {df['Date'].min()} 到 {df['Date'].max()}")
    print(f"股票数量: {df['StockID'].nunique()}")
    print(f"数据列: {list(df.columns)}")
    
    # 检查必要的列是否存在
    required_cols = ['Date', 'StockID', 'Ret_20d']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"缺少必要列: {missing_cols}")
    
    return df

def validate_and_align_data_complete(images_tensor, df, ws=20):
    """
    完整数据验证和对齐，确保所有列保留
    """
    image_samples = images_tensor.shape[0]
    label_samples = len(df)
    
    print("\n" + "="*50)
    print("数据对齐验证")
    print("="*50)
    print(f"图像样本数: {image_samples:,}")
    print(f"标签样本数: {label_samples:,}")
    print(f"标签数据列数: {len(df.columns)}")
    
    if image_samples == label_samples:
        print("✓ 数据样本数匹配")
        return df, image_samples  # 返回原始数据和样本数
    else:
        print("⚠ 数据样本数不匹配，进行对齐处理")
        min_samples = min(image_samples, label_samples)
        
        # 截断图像数据
        images_truncated = images_tensor[:min_samples]
        
        # 截断标签数据但保留所有列
        df_truncated = df.iloc[:min_samples].copy()
        
        print(f"对齐后样本数: {min_samples:,}")
        print(f"对齐后数据形状: {df_truncated.shape}")
        print(f"保留的列: {list(df_truncated.columns)}")
        
        return df_truncated, min_samples

def inference_with_validation(model, test_images, device, batch_size=64):
    """
    带验证的推理函数
    """
    # 检查输入数据有效性
    if torch.isnan(test_images).any():
        nan_count = torch.isnan(test_images).sum().item()
        print(f"警告: 输入图像包含 {nan_count} 个NaN值，进行清理...")
        test_images = torch.nan_to_num(test_images, nan=0.0)
    
    model.eval()
    predictions = []
    
    with torch.no_grad():
        for i in tqdm(range(0, len(test_images), batch_size), desc="推理进度"):
            batch = test_images[i:i+batch_size].to(device)
            outputs = model(batch)
            probs = torch.softmax(outputs, dim=1)
            positive_probs = probs[:, 1].cpu().numpy()
            predictions.extend(positive_probs)
    
    predictions_array = np.array(predictions)
    
    # 验证预测结果
    nan_predictions = np.isnan(predictions_array).sum()
    if nan_predictions > 0:
        print(f"警告: 预测结果包含 {nan_predictions} 个NaN值")
        predictions_array = np.nan_to_num(predictions_array, nan=0.5)  # 将NaN替换为0.5
    
    # 输出统计
    print(f"预测统计:")
    print(f"  均值: {predictions_array.mean():.4f}")
    print(f"  标准差: {predictions_array.std():.4f}")
    print(f"  范围: [{predictions_array.min():.4f}, {predictions_array.max():.4f}]")
    print(f"  >0.5比例: {(predictions_array > 0.5).mean():.2%}")
    
    return predictions_array

def save_complete_results(df, predictions, output_path, prob_col_name='cnn_prob'):
    """
    保存完整结果，确保所有原始列都被保留
    """
    # 创建数据副本以避免修改原始数据
    result_df = df.copy()
    
    # 确保预测数量匹配
    if len(result_df) != len(predictions):
        min_len = min(len(result_df), len(predictions))
        print(f"警告: 最终数据长度不匹配，截断到 {min_len} 个样本")
        result_df = result_df.iloc[:min_len].copy()
        predictions = predictions[:min_len]
    
    # 添加预测列
    result_df[prob_col_name] = predictions
    
    # 确保输出目录存在
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # 验证最终数据完整性
    print(f"最终数据形状: {result_df.shape}")
    print(f"最终数据列: {list(result_df.columns)}")
    print(f"预测列统计: {result_df[prob_col_name].describe()}")
    
    # 保存结果
    result_df.to_feather(output_path)
    print(f"✓ 结果已保存到: {output_path}")
    
    return result_df

def comprehensive_inference_pipeline():
    """
    综合推理流水线 - 确保完整列保留
    """
    # 配置参数
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ws = 20
    print(f"使用设备: {device}")
    print(f"时间窗口: {ws}天")
    
    # 文件路径配置
    model_path = "/workspace_hdd/wangjiang/monthly_20d/exp/checkpoints_filters_128/model_best.pt"
    test_dat_path = "/workspace_hdd/wangjiang/monthly_20d/20d_month_has_vb_[20]_ma_2000_images.dat"
    feather_path = "/workspace_hdd/wangjiang/monthly_20d/20d_month_has_vb_[20]_ma_2000_labels_w_delay.feather"
    output_path = "/workspace_ssd/wangjiang/monthly_20d_cnn_prob/20d_month_has_vb_[20]_ma_2000_labels_w_delay_with_cnn.feather"
    
    print("="*60)
    print("开始完整列保留推理流程")
    print("="*60)
    
    try:
        # 步骤1: 加载预训练模型
        print("\n步骤1: 加载预训练模型")
        model = load_pretrained_model(model_path, ws=ws, device=device)
        
        # 步骤2: 加载图像数据
        print("\n步骤2: 加载图像数据")
        test_images, image_samples = load_test_images_optimized(test_dat_path, ws=ws)
        
        # 步骤3: 完整加载标签数据（保留所有列）
        print("\n步骤3: 加载完整标签数据")
        df = load_labels_complete(feather_path)
        
        # 步骤4: 数据验证和对齐
        print("\n步骤4: 数据验证和对齐")
        aligned_df, final_samples = validate_and_align_data_complete(test_images, df, ws=ws)
        
        # 对齐图像数据
        test_images = test_images[:final_samples]
        
        # 步骤5: 模型推理
        print("\n步骤5: 进行模型推理")
        predictions = inference_with_validation(model, test_images, device, batch_size=128)
        
        # 步骤6: 保存完整结果
        print("\n步骤6: 保存完整结果")
        result_df = save_complete_results(aligned_df, predictions, output_path, 'cnn_prob_60d')
        
        # 步骤7: 最终验证和预览
        print("\n步骤7: 最终验证")
        print("="*50)
        print("处理完成！最终数据验证:")
        print(f"数据形状: {result_df.shape}")
        print(f"数据列数: {len(result_df.columns)}")
        print(f"包含的列: {list(result_df.columns)}")
        
        # 显示完整的数据预览（所有列）
        print("\n完整数据预览（前3行）:")
        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', None)
        print(result_df.head(3))
        
        # 重置显示选项
        pd.reset_option('display.max_columns')
        pd.reset_option('display.width')
        
        # 列统计摘要
        print("\n各列数据类型:")
        print(result_df.dtypes)
        
        print("\n数值列统计摘要:")
        numeric_cols = result_df.select_dtypes(include=[np.number]).columns
        print(result_df[numeric_cols].describe().round(6))
        
        return result_df
        
    except Exception as e:
        print(f"推理过程出错: {e}")
        import traceback
        traceback.print_exc()
        return None

def batch_process_years_with_complete_columns(start_year=2000, end_year=2019, ws=20):
    """
    批量处理多年份数据，确保每一年都完整保留所有列
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    base_data_dir = "/workspace_hdd/wangjiang/monthly_20d"
    output_base_dir = "/workspace_ssd/wangjiang/monthly_20d_cnn_prob"
    
    # 加载模型
    model_path = f"{base_data_dir}/exp/checkpoints_filters_128/model_best.pt"
    model = load_pretrained_model(model_path, ws=ws, device=device)
    
    results = {}
    
    for year in range(start_year, end_year + 1):
        print(f"\n{'='*60}")
        print(f"处理年份: {year}")
        print(f"{'='*60}")
        
        try:
            # 构建文件路径
            dat_path = f"{base_data_dir}/20d_month_has_vb_[20]_ma_{year}_images.dat"
            feather_path = f"{base_data_dir}/20d_month_has_vb_[20]_ma_{year}_labels_w_delay.feather"
            output_path = f"{output_base_dir}/20d_month_has_vb_[20]_ma_{year}_labels_w_delay_with_cnn.feather"
            
            # 检查文件是否存在
            if not all(os.path.exists(p) for p in [dat_path, feather_path]):
                print(f"文件不存在，跳过 {year} 年")
                continue
            
            # 加载数据
            test_images, _ = load_test_images_optimized(dat_path, ws=ws)
            df = load_labels_complete(feather_path)
            
            # 数据对齐
            aligned_df, final_samples = validate_and_align_data_complete(test_images, df, ws=ws)
            test_images = test_images[:final_samples]
            
            # 推理
            predictions = inference_with_validation(model, test_images, device)
            
            # 保存完整结果
            result_df = save_complete_results(aligned_df, predictions, output_path, 'cnn_prob_60d')
            
            # 记录结果
            results[year] = {
                'samples': len(result_df),
                'columns': len(result_df.columns),
                'mean_prob': result_df['cnn_prob_60d'].mean(),
                'file_path': output_path
            }
            
            print(f"✓ 完成 {year} 年处理")
            
        except Exception as e:
            print(f"✗ 处理 {year} 年时出错: {e}")
            continue
    
    # 输出批量处理总结
    print(f"\n{'='*60}")
    print("批量处理总结")
    print(f"{'='*60}")
    for year, info in results.items():
        print(f"{year}年: {info['samples']:,} 样本, {info['columns']} 列, 平均概率: {info['mean_prob']:.4f}")
    
    return results

# 快速验证函数
def quick_validate_output(output_path, expected_columns=None):
    """
    快速验证输出文件的完整性
    """
    if expected_columns is None:
        expected_columns = ['Date', 'StockID', 'MarketCap', 'Ret_5d', 'Ret_20d', 
                           'Ret_60d', 'Ret_month', 'EWMA_vol', 'cnn_prob_60d']
    
    df = pd.read_feather(output_path)
    
    print(f"输出文件验证: {output_path}")
    print(f"数据形状: {df.shape}")
    print(f"包含的列: {list(df.columns)}")
    
    # 检查是否包含所有预期列
    missing_cols = set(expected_columns) - set(df.columns)
    if missing_cols:
        print(f"⚠ 缺少列: {missing_cols}")
    else:
        print("✓ 包含所有预期列")
    
    # 检查数据质量
    print(f"预测列统计:")
    print(df['cnn_prob_60d'].describe())
    
    return df

if __name__ == "__main__":
    # 执行单个文件的推理
    print("开始执行完整列保留推理流程...")
    result_df = comprehensive_inference_pipeline()
    
    if result_df is not None:
        # 快速验证输出
        output_path = "/workspace_ssd/wangjiang/monthly_20d_cnn_prob/20d_month_has_vb_[20]_ma_2000_labels_w_delay_with_cnn.feather"
        quick_validate_output(output_path)
    
    # 批量处理（按需启用）
    # print("\n开始批量处理多年份数据...")
    # batch_results = batch_process_years_with_complete_columns(2000, 2019, 20)
'''

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

def load_year_data(year, base_data_dir, ws=20):
    """加载指定年份的数据"""
    # 构建文件路径
    dat_path = f"{base_data_dir}/20d_month_has_vb_[20]_ma_{year}_images.dat"
    feather_path = f"{base_data_dir}/20d_month_has_vb_[20]_ma_{year}_labels_w_delay.feather"
    
    # 检查文件是否存在
    if not os.path.exists(dat_path) or not os.path.exists(feather_path):
        print(f"⚠  {year}年数据文件不存在，跳过")
        return None, None, None
    
    try:
        # 加载图像数据
        height, width = IMAGE_HEIGHT[ws], IMAGE_WIDTH[ws]
        images_memmap = np.memmap(dat_path, mode='r')
        total_elements = len(images_memmap)
        samples = total_elements // (height * width)
        images = images_memmap[:samples * height * width].reshape(samples, height, width)
        images_tensor = torch.from_numpy(images.copy()).unsqueeze(1).float()
        
        # 加载标签数据（保留所有列）
        df = pd.read_feather(feather_path)
        
        print(f"✓  {year}年数据加载成功: 图像{images.shape}, 标签{df.shape}")
        return images_tensor, df, samples
        
    except Exception as e:
        print(f"✗  {year}年数据加载失败: {e}")
        return None, None, None

def align_data(images_tensor, df, year):
    """对齐图像和标签数据"""
    image_samples = images_tensor.shape[0]
    label_samples = len(df)
    
    if image_samples == label_samples:
        print(f"✓  {year}年数据对齐: {image_samples}样本")
        return images_tensor, df, image_samples
    else:
        min_samples = min(image_samples, label_samples)
        print(f"⚠  {year}年数据不对齐: 图像{image_samples} vs 标签{label_samples}, 使用{min_samples}样本")
        return images_tensor[:min_samples], df.iloc[:min_samples].copy(), min_samples

def inference_year(model, images_tensor, device, batch_size=128, year=2000):
    """对指定年份数据进行推理"""
    model.eval()
    predictions = []
    
    with torch.no_grad():
        for i in tqdm(range(0, len(images_tensor), batch_size), 
                     desc=f"{year}年推理", leave=False):
            batch = images_tensor[i:i+batch_size].to(device, non_blocking=True)
            outputs = model(batch)
            #probs = outputs.argmax(dim=1).cpu().numpy()
            #probs = F.softmax(outputs, dim=1).cpu().numpy()
            #print(probs)
            probs = torch.softmax(outputs, dim=1)[:, 1].cpu().numpy()
            predictions.extend(probs)
    
    predictions = np.array(predictions)
    
    # 处理可能的NaN值
    if np.isnan(predictions).any():
        nan_count = np.isnan(predictions).sum()
        print(f"⚠  {year}年预测包含{nan_count}个NaN，进行清理")
        predictions = np.nan_to_num(predictions, nan=0.5)
    
    print(f"✓  {year}年推理完成: 均值{predictions.mean():.4f}, 范围[{predictions.min():.4f}, {predictions.max():.4f}]")
    return predictions

def save_year_results(df, predictions, output_path, year, prob_col='cnn_prob_60d'):
    """保存年份结果"""
    # 确保数据长度一致
    min_len = min(len(df), len(predictions))
    result_df = df.iloc[:min_len].copy()
    result_df[prob_col] = predictions[:min_len]
    
    # 确保输出目录存在
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # 保存结果
    result_df.to_feather(output_path)
    print(f"✓  {year}年结果保存: {output_path}")
    return result_df

def process_single_year(year, model, base_data_dir, output_base_dir, device, ws=20):
    """处理单个年份的完整流程"""
    print(f"\n{'='*60}")
    print(f"开始处理 {year} 年数据")
    print(f"{'='*60}")
    
    start_time = datetime.now()
    
    try:
        # 1. 加载数据
        images_tensor, df, samples = load_year_data(year, base_data_dir, ws)
        if images_tensor is None or df is None:
            return None
        
        # 2. 数据对齐
        images_aligned, df_aligned, final_samples = align_data(images_tensor, df, year)
        
        # 3. 推理
        predictions = inference_year(model, images_aligned, device, year=year)
        
        # 4. 保存结果
        output_path = f"{output_base_dir}/20d_month_has_vb_[20]_ma_{year}_labels_w_delay_with_cnn.feather"
        result_df = save_year_results(df_aligned, predictions, output_path, year)
        
        processing_time = (datetime.now() - start_time).total_seconds()
        print(f"✓  {year}年处理完成，耗时: {processing_time:.1f}秒")
        
        return {
            'year': year,
            'samples': len(result_df),
            'mean_prob': result_df['cnn_prob_60d'].mean(),
            'file_path': output_path,
            'processing_time': processing_time
        }
        
    except Exception as e:
        print(f"✗  {year}年处理失败: {e}")
        import traceback
        traceback.print_exc()
        return None

def batch_process_years_2000_2019():
    """批量处理2000-2019年数据的主函数"""
    # 配置参数
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ws = 20
    base_data_dir = "/workspace_hdd/wangjiang/monthly_20d"
    output_base_dir = "/workspace_ssd/wangjiang/monthly_60d_cnn_prob_baseline"
    model_path = "/workspace_ssd/wangjiang/monthly_60d/exp/checkpoints_baseline/model_best.pt"
    
    print("="*70)
    print("开始批量处理2000-2019年数据")
    print("="*70)
    print(f"设备: {device}")
    print(f"时间窗口: {ws}天")
    print(f"数据目录: {base_data_dir}")
    print(f"输出目录: {output_base_dir}")
    print(f"模型路径: {model_path}")
    
    # 创建输出目录
    os.makedirs(output_base_dir, exist_ok=True)
    
    # 加载模型（只需一次）
    print("\n1. 加载预训练模型...")
    model = load_pretrained_model(model_path, ws=ws, device=device)
    
    # 处理年份范围
    start_year = 1993
    end_year = 2019
    years = list(range(start_year, end_year + 1))
    
    print(f"\n2. 开始处理{start_year}-{end_year}年数据，共{len(years)}个年份...")
    
    results = {}
    successful_years = 0
    
    # 使用tqdm显示总体进度
    for year in tqdm(years, desc="总体进度"):
        result = process_single_year(year, model, base_data_dir, output_base_dir, device, ws)
        
        if result is not None:
            results[year] = result
            successful_years += 1
    
    # 生成处理报告
    generate_processing_report(results, start_year, end_year, successful_years)
    
    return results

def generate_processing_report(results, start_year, end_year, successful_years):
    """生成处理报告"""
    print("\n" + "="*70)
    print("批量处理完成报告")
    print("="*70)
    
    total_years = end_year - start_year + 1
    success_rate = (successful_years / total_years) * 100
    
    print(f"处理年份: {start_year}-{end_year} (共{total_years}年)")
    print(f"成功处理: {successful_years}年")
    print(f"失败年份: {total_years - successful_years}年")
    print(f"成功率: {success_rate:.1f}%")
    
    if results:
        print(f"\n各年份处理详情:")
        print("-" * 80)
        print(f"{'年份':<6} {'样本数':<10} {'平均概率':<12} {'处理时间(秒)':<15} {'文件路径'}")
        print("-" * 80)
        
        total_samples = 0
        total_time = 0
        
        for year in sorted(results.keys()):
            info = results[year]
            print(f"{year:<6} {info['samples']:<10,} {info['mean_prob']:<12.4f} "
                  f"{info['processing_time']:<15.1f} ...{os.path.basename(info['file_path'])}")
            
            total_samples += info['samples']
            total_time += info['processing_time']
        
        print("-" * 80)
        print(f"{'总计':<6} {total_samples:<10,} {'-':<12} {total_time:<15.1f} {successful_years}个文件")
        
        # 统计信息
        avg_prob = np.mean([info['mean_prob'] for info in results.values()])
        avg_time = total_time / successful_years if successful_years > 0 else 0
        
        print(f"\n统计摘要:")
        print(f"  - 总样本数: {total_samples:,}")
        print(f"  - 平均预测概率: {avg_prob:.4f}")
        print(f"  - 平均处理时间: {avg_time:.1f}秒/年")
        print(f"  - 总处理时间: {total_time:.1f}秒")
        
        # 保存报告到文件
        report_path = f"/workspace_ssd/wangjiang/monthly_20d_cnn_prob/batch_processing_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        save_detailed_report(results, report_path, start_year, end_year, successful_years, 
                           total_samples, avg_prob, total_time)
        
        print(f"\n详细报告已保存至: {report_path}")
    
    print("="*70)

def save_detailed_report(results, report_path, start_year, end_year, successful_years,
                        total_samples, avg_prob, total_time):
    """保存详细处理报告"""
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("CNN模型批量推理处理报告\n")
        f.write("="*50 + "\n")
        f.write(f"处理时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"处理范围: {start_year}-{end_year}年\n")
        f.write(f"成功处理: {successful_years}年\n")
        f.write(f"总样本数: {total_samples:,}\n")
        f.write(f"平均预测概率: {avg_prob:.4f}\n")
        f.write(f"总处理时间: {total_time:.1f}秒\n\n")
        
        f.write("各年份详细结果:\n")
        f.write("-"*80 + "\n")
        f.write(f"{'年份':<6} {'样本数':<10} {'平均概率':<12} {'标准差':<12} {'最小值':<10} {'最大值':<10}\n")
        f.write("-"*80 + "\n")
        
        for year in sorted(results.keys()):
            info = results[year]
            # 这里可以添加更详细的统计信息
            f.write(f"{year:<6} {info['samples']:<10,} {info['mean_prob']:<12.4f} "
                   f"{'-':<12} {'-':<10} {'-':<10}\n")
        
        f.write("-"*80 + "\n")

def validate_batch_results(years_range=None):
    """验证批量处理结果"""
    if years_range is None:
        years_range = range(2000, 2020)
    
    output_base_dir = "/workspace_ssd/wangjiang/monthly_20d_cnn_prob"
    
    print("验证批量处理结果...")
    print("-" * 50)
    
    valid_files = 0
    total_samples = 0
    
    for year in years_range:
        file_path = f"{output_base_dir}/20d_month_has_vb_[20]_ma_{year}_labels_w_delay_with_cnn.feather"
        
        if os.path.exists(file_path):
            try:
                df = pd.read_feather(file_path)
                prob_mean = df['cnn_prob_60d'].mean() if 'cnn_prob_60d' in df.columns else float('nan')
                
                print(f"{year}年: {len(df):>6,}样本, 平均概率: {prob_mean:.4f}")
                valid_files += 1
                total_samples += len(df)
            except Exception as e:
                print(f"{year}年: 文件损坏 - {e}")
        else:
            print(f"{year}年: 文件不存在")
    
    print("-" * 50)
    print(f"有效文件: {valid_files}个")
    print(f"总样本数: {total_samples:,}")
    
    return valid_files, total_samples

# 可选：逐年份处理控制
def process_specific_years(years_list, ws=20):
    """处理指定年份列表"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    base_data_dir = "/workspace_hdd/wangjiang/monthly_20d"
    output_base_dir = "/workspace_ssd/wangjiang/monthly_60d_cnn_prob_baseline"
    model_path = "/workspace_ssd/wangjiang/monthly_60d/exp/checkpoints_baseline/model_best.pt"
    
    model = load_pretrained_model(model_path, ws=ws, device=device)
    
    results = {}
    for year in years_list:
        result = process_single_year(year, model, base_data_dir, output_base_dir, device, ws)
        if result:
            results[year] = result
    
    return results

if __name__ == "__main__":
    # 方案1: 批量处理2000-2019年所有数据
    print("开始执行2000-2019年批量处理...")
    results = batch_process_years_2000_2019()
    
    # 方案2: 验证处理结果
    print("\n开始验证处理结果...")
    valid_files, total_samples = validate_batch_results()
    
    # 方案3: 处理特定年份（如需重新处理某些年份）
    # specific_years = [2000, 2005, 2010, 2015, 2019]  # 示例
    # specific_results = process_specific_years(specific_years)