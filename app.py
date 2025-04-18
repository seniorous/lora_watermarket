import os
import zipfile
import shutil
import json
from flask import Flask, render_template, request, send_file, jsonify
import cv2
import numpy as np
from PIL import Image
from watermark import WatermarkGenerator

app = Flask(__name__)

# 配置上传文件夹
UPLOAD_FOLDER = 'uploads'
PROCESSED_FOLDER = 'processed'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
if not os.path.exists(PROCESSED_FOLDER):
    os.makedirs(PROCESSED_FOLDER)

# 创建水印生成器实例
watermarker = WatermarkGenerator()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/detect_watermark', methods=['POST'])
def detect_watermark():
    if 'image' not in request.files:
        return jsonify({'error': '没有上传文件'}), 400
    
    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': '未选择文件'}), 400

    try:
        # 获取参数
        sensitivity = float(request.form.get('sensitivity', 1.0))
        robust_mode = request.form.get('robust_mode') == 'true'
        
        # 保存原始图片
        input_path = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(input_path)
        
        # 读取图片
        image = cv2.imread(input_path)
        if image is None:
            return jsonify({'error': '无法读取图片文件'}), 400
        
        # 检测水印
        result = watermarker.detect_watermark_robust(image, sensitivity=sensitivity, robust_mode=robust_mode)
        
        # 准备结果页面
        result_html = f"""
        <div class="card mt-4">
            <div class="card-header bg-primary text-white">
                <h5 class="mb-0">水印检测结果</h5>
            </div>
            <div class="card-body">
                <h4 class="{'text-success' if result['has_watermark'] else 'text-danger'}">
                    {'检测到水印' if result['has_watermark'] else '未检测到水印'}
                </h4>
                <p>置信度: {result['confidence']:.4f}</p>
                <p>水印类型: {result['watermark_type']}</p>
                <hr>
                <h5>详细信息:</h5>
                <pre>{json.dumps(result['details'], indent=2, ensure_ascii=False)}</pre>
            </div>
        </div>
        """
        
        return jsonify({
            'success': True,
            'has_watermark': result['has_watermark'],
            'confidence': float(result['confidence']),
            'watermark_type': result['watermark_type'],
            'details': result['details'],
            'result_html': result_html
        })
        
    except Exception as e:
        print(f"检测水印时出错: {str(e)}")
        return jsonify({'error': f'检测水印时出错: {str(e)}'}), 500

@app.route('/process', methods=['POST'])
def process_image():
    if 'image' not in request.files:
        return '没有上传文件', 400
    
    file = request.files['image']
    if file.filename == '':
        return '未选择文件', 400

    try:
        # 获取参数
        visible_text = request.form.get('visible_text', '少游')
        visible_opacity = float(request.form.get('visible_opacity', 0.1))
        invisible_strength = float(request.form.get('invisible_strength', 0.1))
        
        # 保存原始图片
        input_path = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(input_path)
        
        # 读取图片
        image = cv2.imread(input_path)
        if image is None:
            return '无法读取图片文件', 400
        
        # 添加水印
        if request.form.get('add_invisible') == 'true':
            image = watermarker.add_invisible_watermark(image, base_strength=invisible_strength)
        if request.form.get('add_visible') == 'true':
            image = watermarker.add_visible_watermark(image, text=visible_text, opacity=visible_opacity)
        
        # 检测水印强度
        watermark_score = watermarker.detect_watermark(image)
        print(f"水印强度得分: {watermark_score:.4f}")
        
        # 保存处理后的图片
        output_filename = f'watermarked_{file.filename}'
        output_path = os.path.join(PROCESSED_FOLDER, output_filename)
        cv2.imwrite(output_path, image)
        
        # 返回处理后的图片
        return send_file(output_path, as_attachment=True)
        
    except Exception as e:
        print(f"处理图片时出错: {str(e)}")
        return f'处理图片时出错: {str(e)}', 500

@app.route('/process_batch', methods=['POST'])
def process_batch():
    if 'images[]' not in request.files:
        return jsonify({'error': '没有上传文件'}), 400
    
    files = request.files.getlist('images[]')
    if len(files) == 0 or files[0].filename == '':
        return jsonify({'error': '未选择文件'}), 400
    
    # 获取参数
    visible_text = request.form.get('visible_text', '少游')
    visible_opacity = float(request.form.get('visible_opacity', 0.1))
    invisible_strength = float(request.form.get('invisible_strength', 0.1))
    add_invisible = request.form.get('add_invisible') == 'true'
    add_visible = request.form.get('add_visible') == 'true'
    output_dir = request.form.get('output_dir', '')
    
    # 创建临时处理目录
    batch_id = f"batch_{os.urandom(4).hex()}"
    batch_dir = os.path.join(PROCESSED_FOLDER, batch_id)
    os.makedirs(batch_dir, exist_ok=True)
    
    processed_files = []
    failed_files = []
    
    try:
        for file in files:
            try:
                if file.filename == '':
                    continue
                
                # 保存原始图片
                input_path = os.path.join(UPLOAD_FOLDER, file.filename)
                file.save(input_path)
                
                # 读取图片
                image = cv2.imread(input_path)
                if image is None:
                    failed_files.append({'filename': file.filename, 'error': '无法读取图片文件'})
                    continue
                
                # 添加水印
                if add_invisible:
                    image = watermarker.add_invisible_watermark(image, base_strength=invisible_strength)
                if add_visible:
                    image = watermarker.add_visible_watermark(image, text=visible_text, opacity=visible_opacity)
                
                # 保存处理后的图片
                output_filename = f'watermarked_{file.filename}'
                output_path = os.path.join(batch_dir, output_filename)
                cv2.imwrite(output_path, image)
                
                processed_files.append(output_filename)
                
            except Exception as e:
                failed_files.append({'filename': file.filename, 'error': str(e)})
        
        # 创建ZIP文件
        zip_filename = f"{batch_id}.zip"
        zip_path = os.path.join(PROCESSED_FOLDER, zip_filename)
        
        with zipfile.ZipFile(zip_path, 'w') as zipf:
            for filename in processed_files:
                file_path = os.path.join(batch_dir, filename)
                zipf.write(file_path, filename)
        
        # 如果指定了输出目录，复制处理后的文件到该目录
        if output_dir and os.path.isdir(output_dir):
            for filename in processed_files:
                src_path = os.path.join(batch_dir, filename)
                dst_path = os.path.join(output_dir, filename)
                shutil.copy2(src_path, dst_path)
        
        # 清理临时目录
        shutil.rmtree(batch_dir)
        
        return send_file(zip_path, as_attachment=True, download_name=zip_filename)
        
    except Exception as e:
        print(f"批量处理图片时出错: {str(e)}")
        return jsonify({'error': f'批量处理图片时出错: {str(e)}', 'failed_files': failed_files}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)
