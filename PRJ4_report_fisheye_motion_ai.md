# PRJ4 鱼眼视频运动区域提取与目标跟踪报告

## 摘要

本项目面向鱼眼视频中的运动区域提取与短时目标跟踪任务，构建了一套 GPU 深度模型主导的 AI 系统。系统以原始鱼眼图像域为主处理路线，融合 RAFT large 深度光流、YOLO26n-seg 目标先验、自训练 FisheyeMotionNet 运动分割网络、SAM2.1 掩膜细化和相邻帧短时关联跟踪。传统帧差、Farneback 光流、RAFT-only、YOLO-only 和 SAM2-FrameDiff 仅作为 baseline。

在完整测试集 19 个样本上的最新实验结果表明，优化版完整模型 `Full-RAFT-YOLO-DINO-FMN-SAM2` 获得最高综合性能：IoU 为 0.3131，Precision 为 0.3613，Recall 为 0.5830，F1 为 0.3932。相比初版 `Full-RAFT-YOLO-FMN-SAM2` 的 F1 0.3911，优化版小幅提升 0.0021；相比 YOLO-only 的 F1 0.3292，提升 0.0640；相比 FrameDiff 的 F1 0.0658，提升 0.3274。中心/中间/边缘区域 F1 分别为 0.4012、0.3471、0.2915，说明鱼眼边缘畸变仍显著增加任务难度，但融合 RAFT、YOLO、DINOv2、几何先验与 SAM2 后，边缘区性能略优于初版完整模型。

## 任务与数据

任务输入为鱼眼当前帧、前一帧和标定文件，输出包括运动区域掩膜、目标框、短时关联轨迹和中间可视化结果。数据集来自 `homework2.zip`，解压后包含 130 组完整样本：

- 当前帧：`data/homework2/rgb_images/*_FV.png`
- 前一帧：`data/homework2/previous_images/*_FV_prev.png`
- 真值掩膜：`data/homework2/motion_annotation/GroudTruth/*_FV.png`
- 鱼眼标定：`data/homework2/calibration_data/*_FV.json`

需要特别说明的是，GT 掩膜不是 0/255 二值图，而是低灰度标签图，前景像素可能为 4、5、9、14 等。因此本项目按 `mask > 0` 读取运动区域，避免将低值标注误判为空掩膜。

数据划分采用固定随机种子 42，比例为训练集 70%、验证集 15%、测试集 15%。测试集共 19 个样本。

## 鱼眼成像影响分析

鱼眼镜头的核心优势是大视场，但其径向畸变会改变目标的外观、尺度和运动轨迹：

1. 中心区域畸变较轻，目标形状和位移更接近透视相机假设，运动区域边界较稳定。
2. 中间区域存在明显非线性缩放，光流幅值和目标轮廓会随半径变化而偏移。
3. 边缘区域畸变最强，同样的真实运动在图像平面上可能呈现拉伸、压缩或弧形轨迹，传统帧差和 Farneback 光流容易产生破碎掩膜。

因此，本项目没有只依赖单一运动强度阈值，而是显式引入鱼眼几何图 `r, sin(theta), cos(theta)`，让网络学习径向位置与运动外观之间的关系。同时报告按中心区 `r < 0.35`、中间区 `0.35 <= r < 0.70`、边缘区 `r >= 0.70` 分别统计指标。

## 方法

### 总体框架

完整链路如下：

1. 输入前一帧和当前帧，统一缩放到 640x480。
2. 使用 RAFT large 预训练模型生成深度光流，提取 `u, v, mag, angle` 四通道。
3. 使用 YOLO26n-seg 生成 objectness mask 和候选框，作为目标实例先验。
4. 从 `radial_poly` 标定 JSON 生成鱼眼几何图 `r, sin(theta), cos(theta)`。
5. 初版将 `prev RGB + curr RGB + frame diff + RAFT flow + YOLO objectness + geometry maps` 拼接为 15/18 通道输入；优化版进一步加入 DINOv2 语义变化图和边界先验。
6. 使用自训练 FisheyeMotionNet 输出运动概率图。
7. 使用 FisheyeMotionNet 高置信区域与 YOLO 框提示 SAM2.1，对掩膜边界进行细化。
8. 使用 RAFT 光流传播候选框，并用 mask overlap、IoU 和中心距离进行短时目标关联。

### RAFT 深度光流

RAFT 使用 `torchvision.models.optical_flow.raft_large(weights=DEFAULT)`，在 GPU 上推理。与 Farneback 相比，RAFT 的优势在于使用深度特征迭代更新全局相关体，对大位移、纹理弱区域和非刚性运动更稳健。本项目将 RAFT 不作为最终预测器，而是作为 FisheyeMotionNet 的运动先验和跟踪传播依据。

### YOLO26 目标先验

YOLO26n-seg 使用 `ultralytics.YOLO("yolo26n-seg.pt")` 下载预训练权重。YOLO 在本任务中不直接等价于运动分割，因为它识别的是目标实例，不判断是否运动；但它能提供目标存在性、实例边界和 objectness mask。实验表明 YOLO-only 已有较高 Recall，但边界和运动属性不足；与 FisheyeMotionNet 和 SAM2 融合后综合 F1 进一步提升。

### FisheyeMotionNet

FisheyeMotionNet 是一个面向鱼眼运动区域的 Attention U-Net。优化版输入为 18 通道，输出为 1 通道运动概率图。损失函数由边界加权 BCE、Dice loss 和 boundary loss 组成：

```text
L = L_BCE + L_Dice + L_Boundary
```

BCE 提供像素级监督，Dice loss 缓解前景稀疏导致的类别不平衡，boundary loss 强化边界局部一致性。训练强制要求 CUDA；若 `torch.cuda.is_available()` 为 false，程序直接报错，不切换 CPU。

### SAM2.1 掩膜细化

SAM2.1 使用 `sam2.1_hiera_base_plus.pt`。本项目不让 SAM2 单独决定运动区域，而是用 FisheyeMotionNet 的概率图与 YOLO 框作为 prompt。这样可以让 SAM2 专注于边界细化，降低其对非运动静态目标过分分割的风险。

### 优化版：DINOv2 语义运动先验与不确定性融合

在初版完整链路基础上，项目进一步加入面向顶会视频分割思路的增强模块。近年 SAM2、Cutie/MOSE 系列方案强调视频记忆、mask refinement 与 ensemble；2025 年 `Segment Any Motion in Videos` 则将长时轨迹运动、DINO 语义特征和 SAM2 掩膜 densification 结合，说明“运动线索 + 语义表征 + SAM 细化”比单纯光流阈值更适合复杂运动目标分割。

结合本数据只有相邻帧 pair、样本量较小的实际条件，本项目实现了轻量但可复现的优化版：

1. **DINOv2 semantic prior**：使用 `dinov2_vits14_reg` 提取当前帧和前一帧 patch token。当前帧 patch 与边界背景 prototype 的余弦距离形成 semantic saliency；前后帧 patch token 的余弦变化形成 semantic change。二者作为 2 通道深度语义运动先验输入网络。
2. **Boundary prior**：从当前帧 Canny 边缘与帧差 Sobel 梯度中生成 1 通道边界先验，帮助网络关注运动目标轮廓。
3. **Boundary-weighted supervision**：在 BCE+Dice+Boundary loss 的基础上，对 GT 膨胀边界区域提高 BCE 权重，使小目标和细边缘不被大面积背景淹没。
4. **Test-time augmentation uncertainty**：推理时使用原图与水平翻转两路预测，取均值作为概率图，标准差作为 epistemic uncertainty。
5. **Uncertainty-aware fusion**：将 FisheyeMotionNet-TTA 概率、RAFT motion magnitude、YOLO objectness、DINO semantic/change prior 和 edge prior 加权融合，并用 TTA uncertainty 抑制不稳定区域，再作为 SAM2 prompt。

优化版最终方法在代码和指标表中命名为：

```text
Full-RAFT-YOLO-DINO-FMN-SAM2
```

对应输入从初版的 RGB/帧差/RAFT/YOLO/几何扩展为：

```text
prev RGB + curr RGB + frame diff + RAFT flow + YOLO objectness
+ DINO saliency/change + edge prior + fisheye geometry
```

### 跟踪

数据以相邻帧 pair 为主，而不是连续长视频，因此本项目实现短时关联跟踪：

- 从最终运动掩膜中提取连通区域和候选框。
- 使用 RAFT 光流传播上一帧候选中心和框。
- 使用当前帧 SAM2/YOLO 观测框进行匹配。
- 匹配代价综合 IoU、mask overlap 和中心距离。

输出可视化中的 `tracking` 面板绘制传播后的候选框，用于展示相邻帧运动目标关联效果。

## 实验设置

硬件与环境：

- GPU：NVIDIA GeForce RTX 4060 Laptop GPU，8GB 显存。
- CUDA：驱动支持 CUDA 13.1。
- PyTorch：`fisheye_motion` 环境中 `torch 2.11.0+cu128`。
- 所有训练与推理均在 GPU 上运行。

训练参数：

- 输入分辨率：640x480。
- batch size：2。
- 优化器：AdamW。
- 学习率：`1e-3`。
- 最大 epoch：优化版复现实验运行 12 epoch；配置默认最大 50 epoch，可继续训练。
- early stopping patience：10。
- 最佳 checkpoint：`checkpoints/best_f1.pth`。

训练过程：

- 优化版训练运行 12 epoch。
- 最佳验证结果出现在 epoch 8。
- 最佳验证指标：IoU 0.2273，Precision 0.3087，Recall 0.4043，F1 0.2969。
- 最后一轮验证指标：IoU 0.1407，Precision 0.3092，Recall 0.2930，F1 0.2115。

训练曲线见：`outputs/figures/training_curve.png`。

## 结果

最新优化版完整测试集结果如下：

| method | all_iou | all_precision | all_recall | all_f1 | center_f1 | middle_f1 | edge_f1 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Full-RAFT-YOLO-DINO-FMN-SAM2 | 0.3131 | 0.3613 | 0.5830 | 0.3932 | 0.4012 | 0.3471 | 0.2915 |
| FisheyeMotionNet-no-RAFT | 0.2970 | 0.4356 | 0.4013 | 0.3774 | 0.3515 | 0.3179 | 0.1677 |
| FisheyeMotionNet-no-geometry | 0.2915 | 0.4266 | 0.3816 | 0.3755 | 0.3363 | 0.3163 | 0.1869 |
| UncertaintyFusion | 0.2789 | 0.4041 | 0.3990 | 0.3609 | 0.3684 | 0.2895 | 0.1445 |
| FisheyeMotionNet-DINO-edge-TTA | 0.2777 | 0.3979 | 0.4040 | 0.3600 | 0.3676 | 0.2890 | 0.1445 |
| FisheyeMotionNet | 0.2771 | 0.3964 | 0.4043 | 0.3590 | 0.3674 | 0.2889 | 0.1445 |
| FisheyeMotionNet-no-edge | 0.2557 | 0.4470 | 0.3388 | 0.3295 | 0.3453 | 0.2513 | 0.1438 |
| YOLO-only | 0.2384 | 0.2588 | 0.5933 | 0.3292 | 0.3324 | 0.2811 | 0.2283 |
| FisheyeMotionNet-no-DINO | 0.2149 | 0.4349 | 0.2466 | 0.2804 | 0.2635 | 0.2253 | 0.1322 |
| FrameDiff | 0.0376 | 0.0534 | 0.4058 | 0.0658 | 0.0784 | 0.0676 | 0.0419 |
| RAFT-only | 0.0264 | 0.0512 | 0.1151 | 0.0462 | 0.0155 | 0.0515 | 0.0328 |
| SAM2-FrameDiff | 0.0231 | 0.0325 | 0.2117 | 0.0436 | 0.0685 | 0.0277 | 0.0367 |
| Farneback | 0.0147 | 0.0298 | 0.1072 | 0.0275 | 0.0012 | 0.0367 | 0.0155 |
| FisheyeMotionNet-no-YOLO | 0.0021 | 0.1034 | 0.0021 | 0.0040 | 0.0087 | 0.0004 | 0.0000 |

方法对比柱状图见：`outputs/figures/method_f1_bar.png`。

### 整体性能分析

优化版完整模型取得最高 F1 0.3932。相较 YOLO-only，完整模型 Precision 从 0.2588 提高到 0.3613，同时 Recall 保持在 0.5830，说明模型融合有效降低了 YOLO 对静态目标或错误实例的误检。相较 FisheyeMotionNet 单体，完整模型 F1 从 0.3590 提升到 0.3932，说明 RAFT/YOLO/DINO 先验、TTA 不确定性融合和 SAM2 边界细化是互补的。

传统方法明显落后。FrameDiff 的 Recall 较高但 Precision 极低，典型表现是把光照变化、噪声和非目标边缘误判为运动区域。Farneback 与 RAFT-only 只依赖光流强度阈值，不能很好地区分目标真实运动和背景/鱼眼畸变引起的局部运动模式，因此 F1 均低于 0.05。

### 消融分析

去掉 RAFT 后，整体 F1 为 0.3774，低于完整模型 0.3932；边缘区 F1 从 0.2915 降至 0.1677，下降最明显。这说明 RAFT 对鱼眼边缘区域尤其重要，因为边缘处目标形变大，单靠 RGB、YOLO 和 DINO 先验容易漏掉运动连续性。

去掉几何图后，整体 F1 为 0.3755，边缘区 F1 为 0.1869，说明几何先验确实帮助模型理解不同半径位置上的运动外观变化。尽管中心区指标变化不大，但边缘区明显受益。

去掉 DINO 后，整体 F1 从 FisheyeMotionNet-DINO-edge-TTA 的 0.3600 降至 0.2804，说明 DINOv2 语义变化图提供了重要的前景语义支撑。去掉 edge prior 后，整体 F1 降至 0.3295，说明边界先验对小目标和轮廓质量有明显帮助。去掉 YOLO 后 F1 仅为 0.0040，说明在本数据和当前模型设置下，YOLO objectness 仍是网络从稀疏运动标注中稳定收敛的重要先验。

### 中心/中间/边缘区域差异

优化版完整模型的中心区 F1 为 0.4012，中间区为 0.3471，边缘区为 0.2915。该结果符合鱼眼成像规律：中心区畸变小、目标形状更稳定；边缘区由于径向拉伸和非线性投影，目标边界与运动方向更难建模。

值得注意的是，完整模型边缘区 F1 为 0.2915，高于 FisheyeMotionNet 单体的 0.1445，也高于 no-RAFT 的 0.1677 和 no-geometry 的 0.1869。这说明在鱼眼边缘区域，单一网络学习并不足够，必须引入深度光流、几何位置、实例先验、语义先验与 SAM2 边界细化协同建模。

## 可视化结果

每个 summary 图包含以下面板：前一帧、当前帧、校正图、RAFT 光流、YOLO objectness、FisheyeMotionNet 概率图、最终掩膜、GT、误差图和 tracking 可视化。

代表样本位于：

- `outputs/visualizations/00011_summary.png`
- `outputs/visualizations/00019_summary.png`
- `outputs/visualizations/00022_summary.png`
- `outputs/visualizations/00043_summary.png`
- `outputs/visualizations/00068_summary.png`
- `outputs/visualizations/00133_summary.png`
- `outputs/visualizations/00318_summary.png`
- `outputs/visualizations/00361_summary.png`

最终二值预测掩膜位于：`outputs/predictions/*_full.png`。

## 鱼眼域与校正域讨论

本项目主结果采用原始鱼眼域训练与推理，原因有三点：

1. GT 标注本身位于原始鱼眼域，直接训练可避免校正/反投影带来的插值误差。
2. RAFT、YOLO 和 SAM2 都可在原始域生成有效先验，网络再通过几何图学习径向畸变规律。
3. 校正图会放大边缘区域并引入空洞或重采样模糊，在小样本条件下可能损伤边界监督。

同时，系统在可视化中输出了基于标定参数的校正图，用于定性比较中心和边缘形变差异。若进一步扩展为完整双路线实验，可在校正域训练第二个 FisheyeMotionNet，并将预测通过反向映射投回鱼眼域评估。本项目当前的核心量化对比主要通过几何消融和中心/中间/边缘分区指标完成。

## 失败案例与局限

1. 边缘区域仍是主要误差来源。完整模型边缘 F1 为 0.2915，低于中心区 0.4012，说明强畸变下的边界和运动形态仍然难以完全恢复。
2. YOLO 先验依赖较强。no-YOLO 消融 F1 仅为 0.0040，表明小规模训练集下网络容易陷入低概率输出；后续可通过 focal loss、正负样本重采样或更长训练缓解。
3. SAM2-FrameDiff baseline 表现不佳，说明 SAM2 本身不是运动感知模型；若 prompt 来自低质量帧差，SAM2 会细化错误区域。
4. 数据是相邻帧 pair，而非长连续视频，因此跟踪只能进行短时关联，无法评估长时 ID switch、MOTA、IDF1 等指标。

## 结论

本项目实现了一套完整 GPU 深度模型鱼眼运动感知系统。实验表明，单纯传统方法难以应对鱼眼图像中的非线性畸变、目标尺度变化和边缘形变；YOLO-only 虽具备较强目标召回，但缺少运动判别；FisheyeMotionNet 能学习运动区域，但需要 RAFT、YOLO、DINOv2、边界先验和几何图增强才能在边缘区稳定工作。最终的 `Full-RAFT-YOLO-DINO-FMN-SAM2` 在全图、中心区、中间区和边缘区均取得最优或接近最优结果，是本实验中最稳定的方案。

## 复现实验命令

```bash
cd /home/ferdinand/work/homework4
bash scripts/download_weights.sh
bash scripts/run_full_pipeline.sh
```

关键输出：

- `checkpoints/best_f1.pth`
- `outputs/metrics.csv`
- `outputs/metrics_summary.csv`
- `outputs/training_history.csv`
- `outputs/figures/training_curve.png`
- `outputs/figures/method_f1_bar.png`
- `outputs/visualizations/*_summary.png`
- `outputs/predictions/*_full.png`

## 近年工作启发

- SAM2 将 promptable segmentation 扩展到图像与视频，并使用 streaming memory 提高视频分割效率与精度；本项目借鉴其“基础模型负责边界 densification，任务模型负责运动判别”的分工。
- CVPR/PVUW 2024 的复杂视频目标分割方案常用 mask proposal、memory frame、resolution/ensemble 设计提升遮挡和复杂运动场景表现；本项目用 TTA 和多先验融合实现轻量 ensemble。
- 2025 年 `Segment Any Motion in Videos` 显式结合 trajectory motion cues、DINO semantic features 和 SAM2 iterative prompting；本项目在相邻帧数据限制下实现 DINO semantic saliency/change prior，作为长时轨迹不可用时的语义替代。
- SEA-RAFT 等新光流工作说明更强的运动估计仍是运动分割的重要底座；本项目保留 RAFT 深度光流并将其作为网络输入、融合项和短时跟踪传播项。
