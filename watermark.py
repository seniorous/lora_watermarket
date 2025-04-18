import os
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy.fft import dct, idct

class WatermarkGenerator:
    def __init__(self, key_seed=42):
        """
        初始化水印生成器
        key_seed: 用于生成随机水印的种子
        """
        self.key_seed = key_seed
        np.random.seed(key_seed)
        
    def _calculate_image_complexity(self, image):
        """
        计算图像复杂度以自适应调整水印强度
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gradient_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        gradient_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        gradient_magnitude = np.sqrt(gradient_x**2 + gradient_y**2)
        return np.mean(gradient_magnitude)

    def _generate_watermark_pattern(self, shape, strength):
        """
        生成随机水印模式
        """
        return np.random.randn(*shape) * strength

    def add_visible_watermark(self, image, text="少游", opacity=0.1):
        """添加改进的可见水印"""
        if isinstance(image, np.ndarray):
            image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        
        # 创建水印层
        watermark = Image.new('RGBA', image.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(watermark)
        
        # 设置字体
        try:
            font = ImageFont.truetype("STXINGKA.TTF", 72)
        except:
            try:
                font = ImageFont.truetype("simkai.ttf", 72)
            except:
                font = ImageFont.load_default()
        
        # 添加多个位置的水印
        text_width = draw.textlength(text, font=font)
        text_height = 72
        positions = [
            (50, 50),  # 左上
            (image.size[0] - text_width - 50, 50),  # 右上
            (50, image.size[1] - text_height - 50),  # 左下
            (image.size[0] - text_width - 50, image.size[1] - text_height - 50)  # 右下
        ]
        
        for x, y in positions:
            # 添加随机微扰动
            x_offset = np.random.randint(-5, 5)
            y_offset = np.random.randint(-5, 5)
            draw.text((x + x_offset, y + y_offset), text, font=font, 
                     fill=(255, 255, 255, int(255 * opacity)))
        
        watermarked = Image.alpha_composite(image.convert('RGBA'), watermark)
        return cv2.cvtColor(np.array(watermarked.convert('RGB')), cv2.COLOR_RGB2BGR)

    def add_invisible_watermark(self, image, base_strength=0.1):
        """添加改进的不可见水印"""
        # 计算图像复杂度并调整强度
        complexity = self._calculate_image_complexity(image)
        adaptive_strength = base_strength * (1 + np.log1p(complexity) / 10)
        
        # 对每个颜色通道分别处理
        result = image.copy()
        if len(image.shape) == 3:
            for i in range(3):
                channel = image[..., i].astype(np.float32)
                
                # 应用DCT变换
                dct_channel = cv2.dct(channel)
                h, w = dct_channel.shape
                
                # 在多个频率区域添加水印
                regions = [
                    (h//4, h//2, w//4, w//2),    # 中频区域
                    (h//8, h//4, w//8, w//4),    # 低频区域
                    (h//2, 3*h//4, w//2, 3*w//4) # 高频区域
                ]
                
                for h1, h2, w1, w2 in regions:
                    watermark = self._generate_watermark_pattern(
                        (h2-h1, w2-w1), 
                        adaptive_strength
                    )
                    dct_channel[h1:h2, w1:w2] += watermark
                
                # 反DCT变换
                result[..., i] = cv2.idct(dct_channel)
        
        return np.clip(result, 0, 255).astype(np.uint8)

    def detect_watermark(self, image, threshold=0.1):
        """
        检测图像中的水印
        返回水印强度得分
        """
        score = 0
        if len(image.shape) == 3:
            for i in range(3):
                channel = image[..., i].astype(np.float32)
                dct_channel = cv2.dct(channel)
                h, w = dct_channel.shape
                
                # 检查预定义的水印区域
                regions = [
                    (h//4, h//2, w//4, w//2),
                    (h//8, h//4, w//8, w//4),
                    (h//2, 3*h//4, w//2, 3*w//4)
                ]
                
                for h1, h2, w1, w2 in regions:
                    region = dct_channel[h1:h2, w1:w2]
                    # 计算区域能量
                    energy = np.mean(np.abs(region))
                    score += energy
        
        return score / (3 * len(regions))
        
    def _corner_similarity(self, img, size=256):
        """
        计算图像四角区域的相似度，用于检测可见水印
        真正的水印通常在四角有相似的模式
        """
        from skimage.metrics import structural_similarity as ssim
        
        h, w = img.shape[:2]
        # 确保size不超过图像尺寸的1/4
        size = min(size, min(h, w) // 4)
        
        # 提取四个角落的区域
        patches = [
            img[0:size, 0:size],                # 左上
            img[0:size, w-size:w],              # 右上
            img[h-size:h, 0:size],              # 左下
            img[h-size:h, w-size:w]             # 右下
        ]
        
        # 计算6组两两相关性
        cors = []
        for i in range(4):
            for j in range(i+1, 4):
                if len(img.shape) == 3:
                    cors.append(ssim(patches[i], patches[j], channel_axis=2))
                else:
                    cors.append(ssim(patches[i], patches[j]))
        
        return np.array(cors)
        
    def _sigmoid_confidence(self, score, mid, steepness=0.1):
        """
        使用sigmoid函数将得分映射为0-1之间的置信度
        
        参数:
        score: 原始得分
        mid: 中点值（对应0.5置信度）
        steepness: 曲线陡峭程度
        
        返回:
        confidence: 0-1之间的置信度值
        """
        return 1.0 / (1.0 + np.exp(-steepness * (score - mid)))
    
    def _calculate_adaptive_threshold(self, image_complexity, base_threshold=15.0, k=2.0):
        """
        计算自适应阈值，根据图像复杂度动态调整
        
        参数:
        image_complexity: 图像复杂度指标
        base_threshold: 基础阈值
        k: 调整系数
        
        返回:
        threshold: 自适应阈值
        """
        # 根据图像复杂度调整阈值，复杂图像需要更高的阈值
        return base_threshold * (1.0 + np.log1p(image_complexity) / (10.0 * k))
    
    def detect_watermark_robust(self, image, sensitivity=1.0, robust_mode=True, debug=False):
        """
        鲁棒性水印检测算法，能够应对图像缩放、裁剪、压缩等变化
        
        参数:
        image: 输入图像
        sensitivity: 检测灵敏度，值越大越敏感
        robust_mode: 是否启用鲁棒性模式
        
        返回:
        result: 包含检测结果的字典
        """
        result = {
            "has_watermark": False,
            "confidence": 0.0,
            "watermark_type": "none",
            "details": {}
        }
        
        # 预处理图像
        if robust_mode:
            # 标准化图像大小以处理缩放问题
            std_size = (512, 512)
            resized = cv2.resize(image, std_size, interpolation=cv2.INTER_CUBIC)
        else:
            resized = image.copy()
        
        # 转换为灰度图进行分析
        if len(resized.shape) == 3:
            gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
        else:
            gray = resized.copy()
        
        # 计算图像复杂度，用于自适应阈值
        image_complexity = self._calculate_image_complexity(image)
        
        # 应用多尺度分析以提高鲁棒性 - 扩展尺度范围
        scales = [0.5, 0.75, 1.0, 1.25, 1.5] if robust_mode else [1.0]
        max_score = 0
        max_scale = 1.0
        best_dct_image = None
        
        for scale in scales:
            if scale != 1.0:
                current = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
            else:
                current = gray.copy()
            
            # 应用DCT变换
            h, w = current.shape
            # 确保尺寸是8的倍数，DCT效果更好
            h_pad, w_pad = h, w
            if h % 8 != 0:
                h_pad = ((h // 8) + 1) * 8
            if w % 8 != 0:
                w_pad = ((w // 8) + 1) * 8
            
            if h_pad != h or w_pad != w:
                padded = np.zeros((h_pad, w_pad), dtype=np.float32)
                padded[:h, :w] = current
                current = padded
            
            dct_image = cv2.dct(current.astype(np.float32))
            
            # 分析DCT系数的统计特性
            h, w = dct_image.shape
            
            # 检查多个频率区域 - 增加更多区域以提高检测率
            regions = [
                (h//4, h//2, w//4, w//2),      # 中频区域
                (h//8, h//4, w//8, w//4),      # 低频区域
                (h//2, 3*h//4, w//2, 3*w//4),  # 高频区域
                (h//16, h//8, w//16, w//8)     # 更低频区域
            ]
            
            region_scores = []
            for h1, h2, w1, w2 in regions:
                region = dct_image[h1:h2, w1:w2]
                
                # 计算区域能量和统计特性
                energy = np.mean(np.abs(region))
                std_dev = np.std(region)
                entropy = -np.sum(np.abs(region) * np.log2(np.abs(region) + 1e-10))
                
                # 计算区域得分 - 调整权重以提高检测率
                region_score = energy * (1 + std_dev/50) * (1 + entropy/800)
                region_scores.append(region_score)
            
            # 计算总得分
            score = np.mean(region_scores) * sensitivity
            
            if score > max_score:
                max_score = score
                max_scale = scale
                best_dct_image = dct_image
        
        # 应用自适应阈值 - 使用图像复杂度动态调整
        base_threshold = 15.0  # 降低基准阈值，提高检测率
        adaptive_threshold = self._calculate_adaptive_threshold(image_complexity, base_threshold, sensitivity)
        
        # 计算置信度 - 使用sigmoid函数映射
        invisible_confidence = self._sigmoid_confidence(max_score, adaptive_threshold, 0.05)
        
        # 更新结果
        result["details"] = {
            "best_scale": max_scale,
            "threshold": adaptive_threshold,
            "score": max_score,
            "image_complexity": float(image_complexity)
        }
        
        if debug:
            print(f"不可见水印分析: 最大得分={max_score:.4f}, 阈值={adaptive_threshold:.4f}, 置信度={invisible_confidence:.4f}, 最佳尺度={max_scale}")
        
        # 检查DCT系数的分布特性
        if best_dct_image is not None:
            dct_std = np.std(best_dct_image)
            # 计算DCT系数的熵值，用于判断是否存在水印模式
            dct_entropy = -np.sum(np.abs(best_dct_image) * np.log2(np.abs(best_dct_image) + 1e-10))
            result["details"]["dct_std"] = float(dct_std)
            result["details"]["dct_entropy"] = float(dct_entropy)
            
            # 水印检测的关键判断条件 - 使用更灵活的判断标准
            # 1. 置信度必须足够高
            # 2. DCT系数的标准差必须在合理范围内
            # 3. DCT系数的熵值必须足够高，表明存在复杂模式
            if invisible_confidence > 0.6 and dct_std > 20.0 and dct_entropy > 7000.0:
                result["has_watermark"] = True
                result["confidence"] = invisible_confidence
                
                # 尝试确定水印类型
                if invisible_confidence > 0.8:
                    result["watermark_type"] = "strong_invisible"
                else:
                    result["watermark_type"] = "weak_invisible"
        
        # 检测可见水印 - 优先检测不可见水印，如果已经检测到高置信度的不可见水印，则跳过可见水印检测
        if len(image.shape) == 3 and (not result["has_watermark"] or result["confidence"] < 0.75):
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            # 检查饱和度和亮度通道
            s_channel = hsv[:,:,1]
            v_channel = hsv[:,:,2]
            
            # 计算饱和度和亮度的局部变化
            s_grad = cv2.Laplacian(s_channel, cv2.CV_64F)
            v_grad = cv2.Laplacian(v_channel, cv2.CV_64F)
            
            # 计算基准值用于比较
            s_grad_mean = np.mean(np.abs(s_grad))
            v_grad_mean = np.mean(np.abs(v_grad))
            
            # 计算四角相似度，用于检测可见水印
            corner_corrs = self._corner_similarity(image, size=min(image.shape[:2])//4)
            corner_corr_mean = float(np.mean(corner_corrs))
            result["details"]["corner_corr_mean"] = corner_corr_mean
            
            # 使用自适应阈值，根据图像自身特性动态调整
            base_vis = (s_grad_mean + v_grad_mean) * 0.8  # 降低系数，提高检测率
            visible_threshold = max(base_vis, 40.0)  # 降低最小阈值
            
            # 检测可能的水印区域
            visible_score = (s_grad_mean + v_grad_mean) * sensitivity
            
            # 添加额外的统计检查
            s_std = np.std(s_channel)
            v_std = np.std(v_channel)
            
            # 计算局部区域的变化模式，用于判断是否存在可见水印
            # 真正的水印通常会在图像中形成特定的模式
            # 使用更小的块大小，提高检测精度
            block_size = 32
            s_blocks = np.std([np.std(s_channel[i:i+block_size, j:j+block_size]) 
                              for i in range(0, s_channel.shape[0], block_size) 
                              for j in range(0, s_channel.shape[1], block_size) if i+block_size < s_channel.shape[0] and j+block_size < s_channel.shape[1]])
            
            # 保持与s_blocks相同的块大小
            v_blocks = np.std([np.std(v_channel[i:i+block_size, j:j+block_size]) 
                              for i in range(0, v_channel.shape[0], block_size) 
                              for j in range(0, v_channel.shape[1], block_size) if i+block_size < v_channel.shape[0] and j+block_size < v_channel.shape[1]])
            
            # 计算更复杂的模式分数，考虑更多特征
            edges = cv2.Canny(image, 100, 200)
            edge_density = np.sum(edges > 0) / (edges.shape[0] * edges.shape[1])
            
            # 综合评分，考虑边缘密度
            pattern_score = (s_blocks * v_blocks) * (1 + edge_density) * sensitivity
            
            result["details"]["visible_score"] = float(visible_score)
            result["details"]["pattern_score"] = float(pattern_score)
            
            # 计算可见水印置信度 - 使用sigmoid函数映射
            visible_confidence = self._sigmoid_confidence(visible_score, visible_threshold, 0.05)
            result["details"]["visible_confidence"] = float(visible_confidence)
            
            # 可见水印的判断条件 - 使用更灵活的判断标准
            edge_ok = 0.01 < edge_density < 0.30  # 放宽边缘密度范围
            
            has_visible_watermark = (visible_confidence > 0.5 and 
                                   pattern_score > 8.0 and 
                                   s_std > 12.0 and 
                                   edge_ok and
                                   corner_corr_mean > 0.15)  # 放宽角落相似度要求
            
            if has_visible_watermark:
                # 只有当不可见水印未被检测到，或可见水印置信度更高时，才更新结果
                if result["watermark_type"] == "none" or visible_confidence > result["confidence"]:
                    result["has_watermark"] = True
                    result["confidence"] = visible_confidence
                    result["watermark_type"] = "visible"
                elif result["has_watermark"]:
                    # 如果已经检测到不可见水印，且置信度更高，则添加复合类型
                    result["watermark_type"] += "+visible"
        
        return result

def process_images(input_dir, output_dir):
    """处理指定目录下的所有图片"""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    watermarker = WatermarkGenerator()
    
    for filename in os.listdir(input_dir):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(input_dir, filename)
            image = cv2.imread(image_path)
            
            if image is None:
                print(f"无法读取图片: {filename}")
                continue
            
            # 添加水印
            image = watermarker.add_invisible_watermark(image)
            image = watermarker.add_visible_watermark(image)
            
            # 检测水印强度
            watermark_score = watermarker.detect_watermark(image)
            print(f"图片 {filename} 的水印强度得分: {watermark_score:.4f}")
            
            # 保存结果
            output_path = os.path.join(output_dir, f"watermarked_{filename}")
            cv2.imwrite(output_path, image)
            print(f"已处理: {filename}")

def test_watermark_detection(image_path=None):
    """测试水印检测功能"""
    watermarker = WatermarkGenerator()
    
    if image_path is None:
        # 创建一个测试图像
        test_image = np.ones((512, 512, 3), dtype=np.uint8) * 255
        # 添加一些随机纹理
        noise = np.random.randint(0, 50, (512, 512, 3), dtype=np.uint8)
        test_image = cv2.subtract(test_image, noise)
    else:
        # 读取指定图像
        test_image = cv2.imread(image_path)
        if test_image is None:
            print(f"无法读取图像: {image_path}")
            return
    
    # 测试原始图像（应该没有水印）
    print("\n测试原始图像（无水印）:")
    result_original = watermarker.detect_watermark_robust(test_image)
    print(f"检测结果: 有水印={result_original['has_watermark']}, 置信度={result_original['confidence']:.4f}, 类型={result_original['watermark_type']}")
    
    # 添加不可见水印
    invisible_watermarked = watermarker.add_invisible_watermark(test_image.copy(), base_strength=0.2)
    print("\n测试添加不可见水印后的图像:")
    result_invisible = watermarker.detect_watermark_robust(invisible_watermarked)
    print(f"检测结果: 有水印={result_invisible['has_watermark']}, 置信度={result_invisible['confidence']:.4f}, 类型={result_invisible['watermark_type']}")
    
    # 添加可见水印
    visible_watermarked = watermarker.add_visible_watermark(test_image.copy(), opacity=0.2)
    print("\n测试添加可见水印后的图像:")
    result_visible = watermarker.detect_watermark_robust(visible_watermarked)
    print(f"检测结果: 有水印={result_visible['has_watermark']}, 置信度={result_visible['confidence']:.4f}, 类型={result_visible['watermark_type']}")
    
    # 添加两种水印
    both_watermarked = watermarker.add_visible_watermark(invisible_watermarked.copy(), opacity=0.2)
    print("\n测试同时添加两种水印后的图像:")
    result_both = watermarker.detect_watermark_robust(both_watermarked)
    print(f"检测结果: 有水印={result_both['has_watermark']}, 置信度={result_both['confidence']:.4f}, 类型={result_both['watermark_type']}")
    
    return {
        "original": result_original,
        "invisible": result_invisible,
        "visible": result_visible,
        "both": result_both
    }

if __name__ == "__main__":
    # 测试水印检测功能
    test_results = test_watermark_detection()
    
    # 处理图像
    input_dir = "imgs"
    output_dir = "imgs/watermarked"
    process_images(input_dir, output_dir)
