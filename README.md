# LoRA 水印嵌入工具

基于扩散模型的图像水印嵌入工具，用于为图像数据集添加不可见水印，特别适用于训练 LoRA 模型之前对数据集进行预处理。该项目参考了 [Anti-DreamBooth](https://github.com/VinAIResearch/Anti-DreamBooth) 中的攻击算法，将其应用于水印嵌入，创建一个具有特殊标记的"毒"数据集。

## 功能特点

- **水印嵌入**：为图像添加不可见的水印
  - **FSMG (特征空间模式生成)**：通过扩散模型在频域空间添加水印
  - **ASPL (自适应对抗水印)**：通过添加精心设计的水印模式来抵抗模型微调中的水印移除
- **水印检测**：可以检测图像中是否存在水印
- **批量处理**：支持批量处理整个数据集
- **图形界面**：提供友好的图形用户界面
- **详细日志**：生成详细的处理报告和检测结果

## 安装

1. 克隆仓库：

```bash
git clone <repository-url>
cd lora_watermarket
```

2. 安装依赖：

```bash
pip install -r requirements.txt
pip install tk pillow  # 安装GUI所需库
```

## 使用方法

### 使用图形界面（推荐）

在Windows上，直接双击 `run.bat` 文件启动图形界面：

1. 选择输入图像目录
2. 选择输出图像目录
3. 设置水印参数
4. 点击"开始处理"

### 使用命令行

```bash
python main.py --input_dir "输入图像目录" --output_dir "输出图像目录"
```

### 仅检测水印

```bash
python main.py --input_dir "输入图像目录" --output_dir "输出目录" --detect_only
```

### 使用FSMG方法添加水印

```bash
python main.py --input_dir "输入图像目录" --output_dir "输出目录" --method fsmg --watermark_strength 0.1 --gamma 0.6
```

### 使用ASPL方法添加水印

```bash
python main.py --input_dir "输入图像目录" --output_dir "输出目录" --method aspl --watermark_strength 0.1
```

### 所有参数

```
--input_dir         原始输入图像目录
--output_dir        水印图像的输出目录
--model_path        扩散模型路径 (默认: runwayml/stable-diffusion-v1-5)
--method            水印方法: fsmg 或 aspl (默认: fsmg)
--prompt_file       包含每张图像提示词的文本文件，每行一个
--default_prompt    如果没有提供提示词文件，则对所有图像使用此默认提示词
--watermark_strength 水印强度，值越大水印越明显但可能影响图像质量 (默认: 0.1)
--watermark_freq    水印频率，控制水印特征的频率特性 (默认: 0.5)
--gamma             FSMG方法的噪声强度，0.0到1.0之间 (默认: 0.6)
--guidance_scale    无分类器引导比例 (默认: 7.5)
--batch_size        批处理大小 (默认: 1)
--num_steps         扩散或优化步骤数 (默认: 60)
--size              处理图像的大小 (默认: 512)
--seed              随机种子 (默认: 42)
--detect_only       仅检测水印而不添加水印
```

## 版本管理

本项目采用语义化版本管理（Semantic Versioning），版本号格式为：`主版本号.次版本号.修订号`

### 版本号规则

- **主版本号**：当进行不兼容的API更改时增加
- **次版本号**：当增加向下兼容的新功能时增加
- **修订号**：当进行向下兼容的bug修复时增加

### 当前版本

**1.0.0** - 初始版本，实现基本的水印嵌入和检测功能

### 更新日志

| 版本  | 日期       | 更新内容                          |
|-------|------------|-----------------------------------|
| 1.0.0 | 2023-05-01 | 初始版本发布                      |

### 贡献指南

如果您想贡献代码，请遵循以下步骤：

1. Fork本仓库
2. 创建您的特性分支 (`git checkout -b feature/amazing-feature`)
3. 提交您的更改 (`git commit -m 'Add some amazing feature'`)
4. 将更改推送到分支 (`git push origin feature/amazing-feature`)
5. 打开Pull Request

## 工作原理

1. **水印嵌入**：
   - **FSMG方法**：通过扩散模型在频域空间添加水印模式，这些模式对人眼不可见，但可被机器学习模型学习
   - **ASPL方法**：通过对抗性优化添加水印，使其在模型微调过程中能够保持稳定

2. **水印检测**：
   - 使用DCT变换分析图像的频域特征
   - 根据频率特征检测不可见水印的存在
   - 计算图像复杂度动态调整检测阈值

## 使用场景

1. **模型追溯**：通过在训练数据中嵌入水印，可以追踪基于此数据集训练的模型
2. **版权保护**：为自己的图像数据集添加水印，以便识别未经授权使用的衍生模型
3. **数据安全**：在内部数据共享时添加水印以增强数据安全性和可追溯性

## 常见问题解答

### 如何确认水印已成功嵌入？

水印添加后，程序会自动检测并在日志中报告。您也可以使用 `--detect_only` 参数单独检测图像中的水印。

### 水印会影响图像质量吗？

在默认参数下，水印几乎不可见，不会明显影响图像质量。如果增加 `watermark_strength` 参数，水印可能会更明显但也更容易被检测到。

### 使用添加水印的数据集训练的LoRA模型会怎样？

使用添加了水印的数据集训练的LoRA模型，在生成图像时也会带有类似的水印特征，这可以用于追踪和识别模型来源。

## 引用

该项目参考了以下研究成果：

```
@article{datta2024exploiting,
  title={Exploiting Watermark-Based Defense Mechanisms in Text-to-Image Diffusion Models for Unauthorized Data Usage},
  author={Datta, Soumil and Dai, Shih-Chieh and Yu, Leo and Tao, Guanhong},
  journal={arXiv preprint arXiv:2411.15367},
  year={2024}
}
``` 