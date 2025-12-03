# train_from_hdf5.py
import os
import numpy as np
import h5py
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import csv
def run(save_dir,metrics_file,batch_norm,xavier,lrelu,filter_size_list,stride_list,dilation_list,max_pooling_list):
    import os
    import numpy as np
    import h5py
    from tqdm import tqdm
    import torch
    import torch.nn as nn
    from torch.utils.data import Dataset, DataLoader
    import csv
    # ---------------- 配置区 ----------------
    train_h5 = "/workspace_hdd/wangjiang/monthly_20d/train_1993_1999_ori.hdf5"
    val_h5   = "/workspace_hdd/wangjiang/monthly_20d/val_1993_1999_ori.hdf5"
    batch_size = 128
    num_workers = 4#16         # DataLoader 的 num_workers（根据你的机器调整）
    epochs = 50
    lr = 1e-5
    weight_decay = 0#1e-5
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    save_dir =save_dir# "/workspace_ssd/wangjiang/monthly_20d/exp/checkpoints_dropout_075"
    os.makedirs(save_dir, exist_ok=True)

    metrics_file = metrics_file#"./exp/training_log_dropout_075.csv"
    write_header = not os.path.exists(metrics_file)
    if write_header:
        with open(metrics_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["epoch", "train_loss", "train_acc", "val_loss", "val_acc"])

    # ----------------------------------------

    # 你的模型定义（将你原来的 CNNModel 复制进来或从模块 import）
    # 这里假设 CNNModel 已在同一文件或可 import
    # from your_model_file import CNNModel, init_weights

    # 将你之前贴的 CNNModel 代码放在这里或 import
    # 为简洁起见，假设 CNNModel 可用
    # ----------------- 简单示例导入（请确保 CNNModel 在作用域内） -----------------
    # 如果你的 CNNModel 定义在另外的文件，改为： from model_file import CNNModel, init_weights
    # ---------------------------------------------------------------------------

    # ---------- HDF5 Dataset（按需打开 h5 文件，支持多进程） ----------
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

    # 在 worker 终止时确保关闭文件句柄（可选）
    def worker_init_fn(worker_id):
        # nothing for now; Dataset 自己会 lazy-open
        return

    # ---------- 计算 class weights（可选，解决类别不平衡） ----------
    def compute_class_weights(h5_path):
        with h5py.File(h5_path, "r") as f:
            labels = f["labels"][:]
        valid = labels[labels != 2]
        # 计算每类频次
        classes, counts = np.unique(valid, return_counts=True)
        # 如果某个类缺失，确保长度为2
        counts_full = np.zeros(2, dtype=np.float64)
        for c, cnt in zip(classes, counts):
            counts_full[int(c)] = cnt
        total = counts_full.sum()
        # 反频率作为权重（CrossEntropyLoss 期待 weight 为 tensor class_num）
        # 避免 division by zero
        eps = 1e-8
        weights = total / (counts_full + eps)
        # 归一化
        weights = weights / weights.sum() * 2.0
        return torch.tensor(weights, dtype=torch.float32)

    # ----------------- 准备 DataLoader -----------------
    train_dataset = H5Dataset(train_h5)
    val_dataset = H5Dataset(val_h5)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True if device.type == "cuda" else False,
        worker_init_fn=worker_init_fn,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=max(0, num_workers//2),
        pin_memory=True if device.type == "cuda" else False,
        worker_init_fn=worker_init_fn,
    )

    print(f"训练样本数: {len(train_dataset)}, 验证样本数: {len(val_dataset)}")

    # ----------------- 模型实例化 -----------------
    # 你提供的 CNNModel 需要 input_size=(H,W)，layer_number 等超参
    # 这里给出一个合理的默认配置（你可以调整）
    H = train_dataset._h5 or None  # 这里只为说明，实际使用 input_size 对齐下面取自数据
    # 获取图像尺寸（从 h5 头部读取）
    with h5py.File(train_h5, "r") as f:
        H_shape, W_shape = f["images"].shape[1], f["images"].shape[2]

    # 示例超参（根据你的显存、任务自由调整）
    input_size = (H_shape, W_shape)  # (64,60)
    drop_prob = 0.5

    # 如果你的 CNNModel 定义在当前脚本中（替换下方占位），这里直接实例化
    # 请确保 CNNModel 在当前命名空间
    import torch
    import config as cf
    from cnn_model import Model # 直接导入你的模型文件 cnn_model.py


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    ws = 20
    layer_number = cf.BENCHMARK_MODEL_LAYERNUM_DICT[ws]
    filter_size_list, stride_list, dilation_list, max_pooling_list = cf.EMP_CNN_BL_SETTING[ws]
    inplanes = cf.TRUE_DATA_CNN_INPLANES
    #print(filter_size_list)
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
    print(model)
    model_obj.model_summary()
    print("模型参数量:", sum(p.numel() for p in model.parameters()))

    # ----------------- 损失与优化器 -----------------
    # 计算 class weights 并放到 loss（如果你不想使用权重，可以将下面注释掉并使用 nn.CrossEntropyLoss()）
    #class_weights = compute_class_weights(train_h5).to(device)
    #print("class_weights:", class_weights)
    criterion =nn.CrossEntropyLoss()# nn.CrossEntropyLoss(weight=class_weights)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    #scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)

    # ----------------- 训练循环 -----------------
    torch.manual_seed(42)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(42)

    best_val_acc = 0.0
    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        running_correct = 0
        running_total = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch} train", leave=False)
        for imgs, labels in pbar:
            imgs = imgs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad()
            outputs = model(imgs)               # shape (B, num_classes)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * imgs.size(0)
            preds = outputs.argmax(dim=1)
            running_correct += (preds == labels).sum().item()
            running_total += imgs.size(0)

            pbar.set_postfix(loss=running_loss / running_total, acc=running_correct / running_total)

        train_loss = running_loss / running_total
        train_acc = running_correct / running_total

        # 验证
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for imgs, labels in tqdm(val_loader, desc=f"Epoch {epoch} val", leave=False):
                imgs = imgs.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                outputs = model(imgs)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * imgs.size(0)
                preds = outputs.argmax(dim=1)
                val_correct += (preds == labels).sum().item()
                val_total += imgs.size(0)

        val_loss = val_loss / val_total
        val_acc = val_correct / val_total
        with open(metrics_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([epoch, f"{train_loss:.6f}", f"{train_acc:.6f}", f"{val_loss:.6f}", f"{val_acc:.6f}"])



        print(f"Epoch {epoch} | train_loss: {train_loss:.4f} train_acc: {train_acc:.4f} | val_loss: {val_loss:.4f} val_acc: {val_acc:.4f}")

        # 学习率调度
        #scheduler.step(val_loss)

        # 保存最好模型与每 epoch 模型
        torch.save(model.state_dict(), os.path.join(save_dir, f"model_epoch{epoch}.pt"))
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), os.path.join(save_dir, "model_best.pt"))

    print("训练结束。")
    # 关闭 dataset 内部打开的 h5（如果有）
    train_dataset.close()
    val_dataset.close()
run(save_dir="/workspace_ssd/wangjiang/monthly_20d/exp/checkpoints_test",
    metrics_file="./exp/training_log_test.csv",
    batch_norm=True,
    xavier=True,
    lrelu=True,
    filter_size_list= [(5, 3)] * 10,
    stride_list=[(3, 1)] + [(1, 1)] * 10,
    dilation_list=[(2, 1)] + [(1, 1)] * 10,
    max_pooling_list=[(2, 1)] * 10)
'''
run(save_dir="/workspace_ssd/wangjiang/monthly_20d/exp/checkpoints_relu",
    metrics_file="./exp/training_log_relu.csv",
    batch_norm=True,
    xavier=True,
    lrelu=False,
    filter_size_list= [(5, 3)] * 10,
    stride_list=[(3, 1)] + [(1, 1)] * 10,
    dilation_list=[(2, 1)] + [(1, 1)] * 10,
    max_pooling_list=[(2, 1)] * 10)
run(save_dir="/workspace_ssd/wangjiang/monthly_20d/exp/checkpoints_bn_false",
    metrics_file="./exp/training_log_bn_false.csv",
    batch_norm=False,
    xavier=True,
    lrelu=True,
    filter_size_list= [(5, 3)] * 10,
    stride_list=[(3, 1)] + [(1, 1)] * 10,
    dilation_list=[(2, 1)] + [(1, 1)] * 10,
    max_pooling_list=[(2, 1)] * 10)
run(save_dir="/workspace_ssd/wangjiang/monthly_20d/exp/checkpoints_xavier_false",
    metrics_file="./exp/training_log_xavier_false.csv",
    batch_norm=True,
    xavier=False,
    lrelu=True,
    filter_size_list= [(5, 3)] * 10,
    stride_list=[(3, 1)] + [(1, 1)] * 10,
    dilation_list=[(2, 1)] + [(1, 1)] * 10,
    max_pooling_list=[(2, 1)] * 10)

run(save_dir="/workspace_ssd/wangjiang/monthly_20d/exp/checkpoints_max_2_2",
    metrics_file="./exp/training_log_max_2_2.csv",
    batch_norm=True,
    xavier=True,
    lrelu=True,
    filter_size_list= [(5, 3)] * 10,
    stride_list=[(3, 1)] + [(1, 1)] * 10,
    dilation_list=[(2, 1)] + [(1, 1)] * 10,
    max_pooling_list=[(2, 2)] * 10)
run(save_dir="/workspace_ssd/wangjiang/monthly_20d/exp/checkpoints_filter_size_3_3",
    metrics_file="./exp/training_log_filter_size_3_3.csv",
    batch_norm=True,
    xavier=True,
    lrelu=True,
    filter_size_list= [(3, 3)] * 10,
    stride_list=[(3, 1)] + [(1, 1)] * 10,
    dilation_list=[(2, 1)] + [(1, 1)] * 10,
    max_pooling_list=[(2, 1)] * 10)
run(save_dir="/workspace_ssd/wangjiang/monthly_20d/exp/checkpoints_filter_size_7_3",
    metrics_file="./exp/training_log_filter_size_7_3.csv",
    batch_norm=True,
    xavier=True,
    lrelu=True,
    filter_size_list= [(7, 3)] * 10,
    stride_list=[(3, 1)] + [(1, 1)] * 10,
    dilation_list=[(2, 1)] + [(1, 1)] * 10,
    max_pooling_list=[(2, 1)] * 10)


run(save_dir="/workspace_ssd/wangjiang/monthly_20d/exp/checkpoints_ds1",
    metrics_file="./exp/training_log_ds1.csv",
    batch_norm=True,
    xavier=True,
    lrelu=True,
    filter_size_list= [(5, 3)] * 10,
    stride_list=[(1, 1)] + [(1, 1)] * 10,
    dilation_list=[(2, 1)] + [(1, 1)] * 10,
    max_pooling_list=[(2, 1)] * 10)

run(save_dir="/workspace_ssd/wangjiang/monthly_20d/exp/checkpoints_ds2",
    metrics_file="./exp/training_log_ds2.csv",
    batch_norm=True,
    xavier=True,
    lrelu=True,
    filter_size_list= [(5, 3)] * 10,
    stride_list=[(3, 1)] + [(1, 1)] * 10,
    dilation_list=[(1, 1)] + [(1, 1)] * 10,
    max_pooling_list=[(2, 1)] * 10)

run(save_dir="/workspace_ssd/wangjiang/monthly_20d/exp/checkpoints_ds3",
    metrics_file="./exp/training_log_ds3.csv",
    batch_norm=True,
    xavier=True,
    lrelu=True,
    filter_size_list= [(5, 3)] * 10,
    stride_list=[(1, 1)] + [(1, 1)] * 10,
    dilation_list=[(1, 1)] + [(1, 1)] * 10,
    max_pooling_list=[(2, 1)] * 10)
'''