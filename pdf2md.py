import os
import re
import base64
import torch
import traceback
from pdf2image import convert_from_bytes
from io import BytesIO
from threading import Thread
from transformers import (
    AutoProcessor,
    Qwen2_5_VLForConditionalGeneration,
    TextIteratorStreamer
)
from qwen_vl_utils import process_vision_info
from PIL import Image

class StreamPDF2MDConverter:
    def __init__(self, model_name="Qwen2.5-VL-7B-Instruct", device="cuda"):
        # 硬件配置
        self.device = device
        self.fp16 = True if "cuda" in device else False
        
        # 显存优化配置
        self.max_new_tokens = 2000
        self.batch_size = 1  # 可调整批处理大小
        self.dynamic_resolution = True
        
        # 初始化模型
        self._init_model(model_name)
        
        # 编译正则表达式
        self.clean_pattern = re.compile(r'<.*?>|�+')
        self.whitespace_pattern = re.compile(r'\n\s+\n')

    def _init_model(self, model_name):
        """动态加载模型组件，优化显存使用"""
        try:
            # 先加载Processor避免占用过多内存
            self.processor = AutoProcessor.from_pretrained(
                model_name,
                use_fast=True  # 启用快速tokenizer
            )
            
            # 模型加载配置
            model_args = {
                "device_map": "auto" if "cuda" in self.device else None,
                "low_cpu_mem_usage": True,
                # "attn_implementation": "flash_attention_2" if self.fp16 else None
            }
            if self.fp16:
                model_args["torch_dtype"] = torch.float16
            
            # 分阶段加载模型
            with torch.device(self.device):
                self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                    model_name,
                    **model_args
                )
                if self.fp16:
                    self.model = self.model.to(torch.float16)
                
            print(f"模型成功加载到设备: {self.model.device}")
        except Exception as e:
            raise RuntimeError(f"模型初始化失败: {str(e)}")

    def _stream_pdf_pages(self, pdf_input, dpi=150):
        """优化的PDF流式处理"""
        try:
            # 路径验证
            poppler_path = "/usr/bin"
            required_tools = ['pdfinfo', 'pdftocairo']
            for tool in required_tools:
                if not os.path.exists(os.path.join(poppler_path, tool)):
                    raise RuntimeError(f"Poppler工具 {tool} 未找到，请执行: sudo apt-get install poppler-utils")

            # 自动判断输入类型
            if isinstance(pdf_input, str):
                with open(pdf_input, "rb") as f:
                    pdf_bytes = f.read()
            else:
                pdf_bytes = pdf_input

            # 使用生成器逐页转换
            for page_idx, image in enumerate(
                convert_from_bytes(
                    pdf_bytes,
                    dpi=dpi,
                    thread_count=4,
                    fmt="jpeg",
                    use_pdftocairo=True,
                    strict=False,
                    grayscale=True,
                    poppler_path=poppler_path
                )
            ):
                yield page_idx, image
                del image
        except Exception as e:
            raise RuntimeError(f"PDF处理失败: {str(e)}")
        
    def _adjust_dpi_based_on_memory(self, sample_data):
        """根据可用显存动态调整DPI"""
        if not torch.cuda.is_available():
            return 150
        
        free_mem = torch.cuda.mem_get_info()[0] / (1024**3)  # 可用显存(GB)
        if free_mem < 2:
            return 100
        elif free_mem < 4:
            return 120
        else:
            return 150

    def _process_page(self, image):
        """处理单页图像，优化资源管理"""
        try:
            # 图像预处理流水线
            with self._image_pipeline(image) as processed_data:
                # 流式生成配置
                streamer = TextIteratorStreamer(
                    self.processor.tokenizer,
                    timeout=60.0,
                    skip_prompt=True,
                    skip_special_tokens=True
                )
                
                # 异步生成
                generation_thread = Thread(
                    target=self._generate_text,
                    args=(processed_data, streamer)
                )
                generation_thread.start()
                
                # 流式收集结果
                full_text = ""
                for new_text in streamer:
                    full_text += new_text
                    yield full_text  # 支持实时输出

        except Exception as e:
            self._handle_error(e)
            return ""
        finally:
            self._cleanup_resources(image)

    def _image_pipeline(self, image):
        """图像处理上下文管理器"""
        class PipelineWrapper:
            def __init__(self, processor):
                self.processor = processor
                self.device = device
                
            def __enter__(self):
                # 图像编码
                with BytesIO() as buffer:
                    try:
                        image.convert("RGB").save(buffer, format="JPEG", quality=85)
                        img_base64 = base64.b64encode(buffer.getvalue()).decode()
                        self.image_uri = f"data:image/jpeg;base64,{img_base64}"
                    except Exception as e:
                        raise ValueError(f"图像编码失败: {str(e)}")
                
                # 构建符合Qwen要求的输入
                self.messages = [{
                    "role": "user",
                    "content": [
                        {"type": "image", "image": self.image_uri},
                        {"type": "text", "text": (
                            "[系统指令] 严格按Markdown格式输出：\n"
                            "- 保留标题层级结构\n"
                            "- 表格使用管道符\n"
                            "- 代码块用```包裹\n"
                            "- 直接输出结果不要解释"
                        )}
                    ]
                }]
                
                # 处理视觉输入
                try:
                    image_inputs, _ = process_vision_info(self.messages)
                    self.flat_images = [
                        img for item in image_inputs
                        for img in (item if isinstance(item, list) else [item])
                        if isinstance(img, Image.Image)
                    ]
                    assert self.flat_images, "无有效图像输入"
                except Exception as e:
                    raise RuntimeError(f"视觉处理失败: {str(e)}")
                
                # 应用聊天模板
                self.text_prompt = self.processor.apply_chat_template(
                    self.messages,
                    tokenize=False,
                    add_generation_prompt=True
                )
                
                # 准备模型输入
                self.inputs = self.processor(
                    text=[self.text_prompt],
                    images=self.flat_images,
                    padding=True,
                    return_tensors="pt"
                ).to(self.device)
                
                return self.inputs
                
            def __exit__(self, exc_type, exc_val, exc_tb):
                # 显式释放资源
                del self.inputs, self.flat_images, self.messages
                torch.cuda.empty_cache()
                
        return PipelineWrapper(self.processor)

    def _generate_text(self, inputs, streamer):
        """优化的生成过程"""
        with torch.inference_mode():
            try:
                self.model.generate(
                    **inputs,
                    max_new_tokens=self.max_new_tokens,
                    temperature=0.1,
                    repetition_penalty=1.2,
                    do_sample=False,
                    streamer=streamer,
                    pad_token_id=self.processor.tokenizer.pad_token_id
                )
            except Exception as e:
                streamer.put(f"生成错误: {str(e)}")
                streamer.end()

    def _clean_text(self, text):
        """优化的后处理"""
        text = self.clean_pattern.sub('', text)
        return self.whitespace_pattern.sub('\n\n', text).strip()

    def _handle_error(self, error):
        """统一错误处理"""
        error_msg = f"页面处理失败: {str(error)}\n{traceback.format_exc()}"
        print(error_msg)
        if "CUDA out of memory" in str(error):
            print("检测到显存不足，尝试以下措施：")
            print("1. 降低PDF分辨率（设置dpi=100）")
            print("2. 减少batch_size参数")
            print("3. 使用CPU模式运行")

    def _cleanup_resources(self, image):
        """资源清理"""
        try:
            del image
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
        except Exception as e:
            print(f"资源清理失败: {str(e)}")

    def convert(self, pdf_input, output_md="output.md"):
        """最终结果保留方案"""
        try:
            with open(output_md, "w", encoding="utf-8") as md_file:
                for page_num, image in self._stream_pdf_pages(pdf_input):
                    try:
                        print(f"正在处理第 {page_num+1} 页...")
                        
                        # 仅保留最终识别结果
                        final_text = ""
                        for partial_text in self._process_page(image):
                            if partial_text:  # 始终用最新结果覆盖
                                final_text = partial_text
                        
                        # 直接写入最终结果
                        md_file.write(f"<!-- Page {page_num+1} -->\n")
                        md_file.write(final_text + "\n\n")
                    
                    finally:
                        if hasattr(image, 'close'):
                            image.close()
                            
        except Exception as e:
            self._handle_error(e)
            raise
        finally:
            self._cleanup_resources(None)
            print(f"转换完成！输出文件: {output_md}")   

if __name__ == "__main__":
    # 初始化配置
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"运行设备: {device.upper()}")
    
    try:
        converter = StreamPDF2MDConverter(
            model_name="~/models/Qwen/Qwen2.5-VL-7B-Instruct",
            device=device
        )
        
        # 示例文件处理
        converter.convert(
            "/home/test/ocr/pdf/pwn笔记.pdf",
            output_md="/home/test/ocr/markdown/pwn笔记.md"
        )
    except Exception as e:
        print(f"致命错误: {str(e)}")
        exit(1)
