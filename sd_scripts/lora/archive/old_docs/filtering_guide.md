# 训练数据质量筛选流程 (Data Quality Filtering)

## 概述

本流程用于全量筛选 LoRA 训练数据，确保用于训练的样本具有高质量的 mask 标注和清晰的脏污外观。

---

## 工作流程

```
原始 manifest (manifest_train.csv)
         ↓
   交互式质量筛选
         ↓
   filtered + rejected manifests
         ↓
   使用 filtered manifest 重新生成训练数据
         ↓
   高质量训练数据 → LoRA 训练
```

---

## Phase 1: 交互式质量筛选

### 选择筛选工具版本

| 环境 | 推荐工具 | 说明 |
|------|---------|------|
| 有图形界面 (本地/Linux Desktop) | `filter_training_data.py` | GUI 窗口 + 按钮 |
| WSL / 无图形界面 | `filter_training_data_cli.py` | 命令行 + 预览图 |

### WSL 环境 (CLI 版本)

```bash
cd /home/yf/soiling_project/sd_scripts/lora

python filter_training_data_cli.py \
    --manifest training_data/dreambooth_format/manifest_train.csv \
    --output training_data/filter_results \
    --batch-size 20
```

**工作流程**：
1. 脚本生成一批预览图（默认 20 张）到 `filter_results/previews/`
2. 在图像查看器中打开预览图
3. 根据预览图输入命令：
   - `a <file_id>` - 接受样本
   - `r <file_id>` - 拒绝样本
   - `ra` - 接受批次内全部
   - `rr` - 拒绝批次内全部
   - `s` - 跳过此批次
   - `q` - 退出并保存

### 图形界面环境 (GUI 版本)

```bash
cd /home/yf/soiling_project/sd_scripts/lora

python filter_training_data.py \
    --manifest training_data/dreambooth_format/manifest_train.csv \
    --output training_data/filter_results \
    --save-interval 10
```

### 操作说明

#### GUI 版本操作

| 按键 | 功能 |
|------|------|
| **A** | Accept - 接受样本（质量合格） |
| **R** | Reject - 拒绝样本（质量问题） |
| **S** | Skip - 跳过（稍后再审） |
| **← →** | 左右箭头 - 导航浏览 |
| **Q / ESC** | 退出并保存进度 |

#### CLI 版本操作

| 命令 | 功能 |
|------|------|
| `a <file_id>` | 接受指定样本 |
| `r <file_id>` | 拒绝指定样本 |
| `ra` | 接受批次内全部 |
| `rr` | 拒绝批次内全部 |
| `s` | 跳过此批次 |
| `q` | 退出并保存进度 |

### 拒绝原因分类

当选择 **R (Reject)** 时，系统会询问拒绝原因：

| 代码 | 原因 | 说明 |
|------|------|------|
| 1 | mask_inaccurate | 红色脏污区域不准确 |
| 2 | poor_quality | 图像质量差（模糊、噪声） |
| 3 | no_dirt | 无可见脏污 |
| 4 | other | 其他问题 |

### 断点续传

筛选工具会自动保存进度到 `filter_progress.json`。

- 每 N 个样本自动保存（默认 10 个）
- 按 Q 退出时保存
- 重新运行时自动加载进度，从上次位置继续

---

## Phase 2: 重新生成训练数据

筛选完成后，使用筛选结果重新生成训练数据：

```bash
python 02_generate_filtered_training_data.py \
    --filtered-manifest ../training_data/filter_results/manifest_filtered.csv \
    --output ../training_data/dreambooth_format_filtered \
    --bg-method random \
    --seed 42
```

### 输出结构

```
dreambooth_format_filtered/
├── train/
│   ├── 0001_xxxx.jpg
│   ├── 0001_xxxx.txt
│   ├── 0002_xxxx.jpg
│   ├── 0002_xxxx.txt
│   └── ...
└── manifest_train.csv
```

---

## 质量检查标准

### Accept 标准（必须满足）

1. **Mask 准确性**
   - 红色覆盖区域与实际脏污区域高度重合
   - 脏污边缘与 mask 边缘基本吻合
   - 无明显的漏标或误标

2. **图像质量**
   - 图像清晰，无过度模糊
   - 无严重的压缩伪影或噪声
   - 脏污纹理清晰可见

3. **脏污可见性**
   - mask 标记的区域内有明显的脏污特征
   - 脏污与背景有足够的对比度

### Reject 标准（任一情况即拒绝）

1. **Mask 问题**
   - 红色区域明显偏离实际脏污
   - 大量漏标（实际脏污未被标记）
   - 大量误标（清洁区域被标记）

2. **图像问题**
   - 图像过度模糊，脏污细节丢失
   - 严重的压缩块或噪声
   - 曝光异常（过暗/过亮）

3. **标注问题**
   - 标记区域内无可见脏污
   - 脏污类型与标注类别不符

---

## 输出文件说明

### 筛选阶段输出

```
filter_results/
├── filter_progress.json      # 进度文件（断点续传）
├── manifest_filtered.csv     # 通过筛选的样本清单
└── manifest_rejected.csv     # 被拒绝的样本清单
```

### 重新生成阶段输出

```
dreambooth_format_filtered/
├── train/                    # 训练样本
│   ├── {file_id}.jpg        # 训练图像
│   └── {file_id}.txt        # Caption 文本
└── manifest_train.csv       # 训练 manifest
```

---

## 预期工作量估算

| 阶段 | 样本数 | 预计通过率 | 人工工作量 |
|------|--------|-----------|-----------|
| Phase 2a (当前) | 100 | 60-80% | ~20-30 分钟 |
| Phase 2b (完整) | 3000 | 60-80% | ~10-15 小时 |

建议：
- Phase 2a 完成后评估通过率
- 如果通过率 < 60%，建议检查筛选标准是否过严
- 可以分多次完成，利用断点续传功能

---

## 质量报告示例

筛选完成后会生成如下报告：

```
============================================================
Filtering Report
============================================================
Total samples: 100
Reviewed: 100
Accepted: 75 (75.0%)
Rejected: 25 (25.0%)

Rejection reasons:
  mask_inaccurate: 18
  poor_quality: 5
  no_dirt: 2

Filtered manifest saved: .../manifest_filtered.csv
Filtered samples: 75
============================================================
```

---

## 常见问题

### Q: 如果误操作 Accept/Reject 了怎么办？
A: 当前版本不支持撤销，但可以手动编辑 `manifest_filtered.csv` 或 `manifest_rejected.csv`，移动相应的 file_id。

### Q: 筛选过程中途退出会丢失进度吗？
A: 不会。进度会自动保存到 `filter_progress.json`，重新运行时会从上次位置继续。

### Q: 可以同时运行多个筛选会话吗？
A: 不建议。每个 manifest 目录对应一个进度文件，多线程可能导致冲突。

### Q: 如何查看被拒绝的样本？
A: 使用可视化工具查看 `manifest_rejected.csv`：
```bash
python visualize_training_data.py \
    --manifest ../training_data/filter_results/manifest_rejected.csv \
    --output ../training_data/filter_results/rejected_visualization \
    --num-samples 20
```

---

## 下一步

筛选完成后，使用 `dreambooth_format_filtered/` 目录中的数据进行 LoRA 训练：

```bash
# 更新训练脚本路径
# --instance_data_dir 指向 dreambooth_format_filtered/train/
```
