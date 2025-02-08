import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import cv2
import numpy as np
from collections import deque
import random
import os
from glob import glob
from tqdm import tqdm
import json
import hashlib
from datetime import datetime
import matplotlib.pyplot as plt
from PIL import Image, ImageEnhance
import pyautogui
import keyboard
import threading
import queue
from torchvision import transforms
import time
import win32gui
import win32con
import gc
import warnings
import shutil
warnings.filterwarnings('ignore')

import tkinter as tk
from tkinter import messagebox

############################################################
# 全局配置
############################################################
CONFIG = {
    'DEVICE': torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
    'BATCH_SIZE': 32,
    'SEQUENCE_LENGTH': 32,
    'FRAME_SIZE': (256, 256),
    'NUM_WORKERS': min(4, os.cpu_count()),
    'GPU_MEMORY_FRACTION': 0.8,
    'ENABLE_CUDA_OPTIMIZATION': True
}

############################################################
# 透明日志窗口类
############################################################
class OverlayLogger:
   """
   透明窗口日志显示器。
   在屏幕左下角打印实时日志，并使用透明窗口覆盖在最上层。
   """
   def __init__(self, width=600, height=400):
       self.root = None
       self.width = width
       self.height = height
       self.log_queue = queue.Queue()
       self.update_interval = 100  # ms
       self.stop_event = threading.Event()
       self.thread = threading.Thread(target=self._run_overlay, daemon=True)

   def start(self):
       """启动日志窗口线程"""
       self.thread.start()

   def stop(self):
       """停止日志窗口"""
       self.stop_event.set()
       if self.root is not None:
           try:
               self.root.quit()
           except:
               pass
       self.thread.join()

   def add_log(self, text):
       """
       向队列中添加一条日志。
       Args:
           text: 日志文本内容
       """
       self.log_queue.put(text)

   def clear_log(self):
       """清空日志显示"""
       if hasattr(self, 'text_box'):
           self.text_box.delete(1.0, tk.END)

   def _run_overlay(self):
       """
       启动一个 Tk 窗口，显示透明顶层日志。
       """
       self.root = tk.Tk()
       self.root.title("场景状态监测")
       self.root.attributes("-topmost", True)
       self.root.overrideredirect(True)

       # 计算左下角位置
       screen_width = self.root.winfo_screenwidth()
       screen_height = self.root.winfo_screenheight()
       x_pos = 0
       y_pos = screen_height - self.height

       self.root.geometry(f"{self.width}x{self.height}+{x_pos}+{y_pos}")
       self.root.wm_attributes("-alpha", 0.8)

       # 主框架
       main_frame = tk.Frame(self.root, bg='black')
       main_frame.pack(fill=tk.BOTH, expand=True)

       # 标题栏
       title_frame = tk.Frame(main_frame, bg='#1a1a1a', height=30)
       title_frame.pack(fill=tk.X)
       title_frame.pack_propagate(False)

       title_label = tk.Label(
           title_frame, 
           text="实时场景监测", 
           bg='#1a1a1a',
           fg='white',
           font=("Consolas", 10, "bold")
       )
       title_label.pack(side=tk.LEFT, padx=10)

       # 文本框
       self.text_box = tk.Text(
           main_frame,
           bg="black",
           fg="white",
           font=("Consolas", 10),
           padx=10,
           pady=10,
           wrap=tk.WORD,
           relief=tk.FLAT
       )
       self.text_box.pack(fill=tk.BOTH, expand=True)

       # 滚动条
       scrollbar = tk.Scrollbar(self.text_box)
       scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
       self.text_box.config(yscrollcommand=scrollbar.set)
       scrollbar.config(command=self.text_box.yview)

       # 设置文本标签样式
       self.text_box.tag_configure("title", foreground="#00ff00")  # 绿色标题
       self.text_box.tag_configure("warning", foreground="#ff6b6b")  # 红色警告
       self.text_box.tag_configure("info", foreground="#4dc4ff")    # 蓝色信息
       self.text_box.tag_configure("highlight", foreground="#ffcb6b")  # 黄色高亮
       self.text_box.tag_configure("success", foreground="#69f0ae")   # 绿色成功

       # 定时更新日志
       self._update_logs()
       
       # 绑定右键菜单
       self._create_context_menu()
       
       self.root.mainloop()

   def _create_context_menu(self):
       """创建右键菜单"""
       self.context_menu = tk.Menu(self.root, tearoff=0, bg='#2d2d2d', fg='white')
       self.context_menu.add_command(label="清除日志", command=self.clear_log)
       self.context_menu.add_separator()
       self.context_menu.add_command(label="关闭窗口", command=self.stop)
       
       def show_menu(event):
           self.context_menu.post(event.x_root, event.y_root)
       
       self.text_box.bind("<Button-3>", show_menu)

   def _update_logs(self):
       """
       周期性从队列中取日志并显示。
       """
       try:
           while not self.log_queue.empty():
               log_text = self.log_queue.get_nowait()
               
               # 根据日志内容添加不同的样式
               if "错误" in log_text or "警告" in log_text:
                   self.text_box.insert(tk.END, log_text + "\n", "warning")
               elif "成功" in log_text or "完成" in log_text:
                   self.text_box.insert(tk.END, log_text + "\n", "success")
               elif "===" in log_text:
                   self.text_box.insert(tk.END, log_text + "\n", "title")
               elif ":" in log_text:
                   self.text_box.insert(tk.END, log_text + "\n", "info")
               else:
                   self.text_box.insert(tk.END, log_text + "\n")
                   
               self.text_box.see(tk.END)
               
               # 限制文本框内容长度
               if float(self.text_box.index('end')) > 1000:  # 最多保留1000行
                   self.text_box.delete(1.0, 2.0)
       except Exception as e:
           print(f"更新日志出错: {str(e)}")

       if not self.stop_event.is_set():
           self.root.after(self.update_interval, self._update_logs)

############################################################
# 基础环境设置
############################################################
def setup_environment():
    print("\n正在设置运行环境...")
    
    if torch.cuda.is_available():
        print(f"检测到CUDA设备: {torch.cuda.get_device_name(0)}")
        print(f"CUDA版本: {torch.version.cuda}")
        if CONFIG['ENABLE_CUDA_OPTIMIZATION']:
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.enabled = True
            print("CUDA优化已启用")
        try:
            torch.cuda.set_per_process_memory_fraction(
                CONFIG['GPU_MEMORY_FRACTION'], 0
            )
            print(f"GPU内存限制设置为: {CONFIG['GPU_MEMORY_FRACTION']*100:.0f}%")
        except Exception as e:
            print(f"设置GPU内存限制失败: {str(e)}")
    else:
        print("警告: 未检测到CUDA设备，将使用CPU运行")
        CONFIG['DEVICE'] = torch.device('cpu')
    
    required_dirs = ['./videos', './models', './visualizations']
    for dir_path in required_dirs:
        os.makedirs(dir_path, exist_ok=True)
    print("目录结构已创建")
    
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    print("随机种子已设置")
    
    print("环境设置完成\n")

############################################################
# 性能监控器
############################################################
class PerformanceMonitor:
    def __init__(self):
        self.start_time = time.time()
        self.checkpoints = {}
        self.memory_usage = []
    
    def checkpoint(self, name):
        self.checkpoints[name] = time.time() - self.start_time
        if torch.cuda.is_available():
            memory_used = torch.cuda.memory_allocated() / 1024**2
            self.memory_usage.append((name, memory_used))
    
    def report(self):
        print("\n性能报告:")
        print("-" * 50)
        for name, time_taken in self.checkpoints.items():
            print(f"{name}: {time_taken:.2f} 秒")
        if self.memory_usage:
            print("\nGPU内存使用:")
            for name, memory in self.memory_usage:
                print(f"{name}: {memory:.2f} MB")
        print("-" * 50)

monitor = PerformanceMonitor()

############################################################
# 资源清理
############################################################
def cleanup():
   """
   清理资源和临时文件:
   - 清空GPU缓存
   - 关闭OpenCV窗口
   - 删除tmp目录及其内容
   - 执行垃圾回收
   """
   print("\n清理资源...")
   
   # 清理GPU缓存
   if torch.cuda.is_available():
       torch.cuda.empty_cache()
       print("GPU缓存已清理")
   
   # 关闭OpenCV窗口
   cv2.destroyAllWindows()
   print("OpenCV窗口已关闭")
   
   # 删除tmp目录
   tmp_dir = "./tmp"
   sequences_dir = "./tmp/sequences"
   try:
       if os.path.exists(sequences_dir):
           shutil.rmtree(sequences_dir)
           print("序列缓存目录已删除")
       if os.path.exists(tmp_dir):
           shutil.rmtree(tmp_dir)
           print("临时目录已删除") 
   except Exception as e:
       print(f"删除临时文件失败: {str(e)}")
       
   # 垃圾回收
   gc.collect()
   print("垃圾回收已执行")
   print("资源清理完成")

############################################################
# GPU 场景检测器
############################################################
class GPUSceneDetector:
    def __init__(self, batch_size=32):
        print("初始化GPU场景检测器...")
        self.device = CONFIG['DEVICE']
        self.batch_size = batch_size
        self.use_cuda_flow = False
        self.last_frames = None
        
        try:
            if hasattr(cv2.cuda, 'FarnebackOpticalFlow'):
                self.flow_computer = cv2.cuda.FarnebackOpticalFlow.create(
                    numLevels=5,
                    pyrScale=0.5,
                    fastPyramids=False,
                    winSize=21,
                    numIters=3,
                    polyN=7,
                    polySigma=1.5,
                    flags=0
                )
                self.use_cuda_flow = True
                print("使用CUDA光流计算")
            else:
                print("CUDA光流不可用，使用CPU光流计算")
        except Exception as e:
            print(f"初始化CUDA光流失败: {str(e)}")
            print("使用CPU光流计算")

        self.feature_extractor = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        ).to(self.device)
        
        self.scene_classifier = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(256, 8)
        ).to(self.device)
        
        self.motion_classifier = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(256, 7)
        ).to(self.device)
        
        print("GPU场景检测器初始化完成")
    
    @torch.no_grad()
    def process_batch(self, frames):
        try:
            frame_tensors = []
            for frame in frames:
                if isinstance(frame, np.ndarray):
                    frame = torch.from_numpy(frame).permute(2, 0, 1).float() / 255.0
                frame_tensors.append(frame)
            
            batch = torch.stack(frame_tensors).to(self.device)
            batch_size = batch.size(0)
            
            with torch.cuda.amp.autocast():
                features = self.feature_extractor(batch)
                
                scene_logits = self.scene_classifier(features)
                scene_probs = torch.sigmoid(scene_logits)
                
                motion_logits = self.motion_classifier(features)
                motion_probs = torch.sigmoid(motion_logits)
                
                flows = self._compute_batch_flow(batch)
            
            results = []
            for i in range(batch_size):
                scene_info = {
                    'is_menu': scene_probs[i, 0].item() > 0.5,
                    'is_map': scene_probs[i, 1].item() > 0.5,
                    'is_dialog': scene_probs[i, 2].item() > 0.5,
                    'is_loading': scene_probs[i, 3].item() > 0.5,
                    'is_combat': scene_probs[i, 4].item() > 0.5,
                    'is_swimming': scene_probs[i, 5].item() > 0.5,
                    'is_flying': scene_probs[i, 6].item() > 0.5,
                    'is_climbing': scene_probs[i, 7].item() > 0.5,
                    
                    'is_walking': motion_probs[i, 0].item() > 0.5,
                    'is_running': motion_probs[i, 1].item() > 0.5,
                    'is_jumping': motion_probs[i, 2].item() > 0.5,
                    'is_fighting': motion_probs[i, 3].item() > 0.5,
                    'is_swimming_motion': motion_probs[i, 4].item() > 0.5,
                    'is_flying_motion': motion_probs[i, 5].item() > 0.5,
                    'is_climbing_motion': motion_probs[i, 6].item() > 0.5,
                }
                
                if flows is not None and i < len(flows):
                    flow = flows[i]
                    scene_info.update({
                        'has_motion': flow.abs().mean().item() > 0.1,
                        'movement_direction': self._get_direction_from_flow(flow),
                        'vertical_speed': flow[..., 1].mean().item(),
                        'horizontal_speed': flow[..., 0].mean().item()
                    })
                
                results.append(scene_info)
            
            self.last_frames = batch.clone()
            return results
            
        except Exception as e:
            print(f"批处理错误: {str(e)}")
            import traceback
            traceback.print_exc()
            return [{}] * len(frames)
    
    def _compute_batch_flow(self, batch):
        try:
            if self.last_frames is None:
                self.last_frames = batch
                return None
            
            flows = []
            for i in range(len(batch)):
                flow = self._compute_flow(
                    self.last_frames[i],
                    batch[i]
                )
                flows.append(flow)
            
            return flows
            
        except Exception as e:
            print(f"批量光流计算错误: {str(e)}")
            return None
    
    def _compute_flow(self, frame1, frame2):
        try:
            gray1 = frame1.mean(dim=0).cpu().numpy()
            gray2 = frame2.mean(dim=0).cpu().numpy()
            gray1 = (gray1 * 255).astype(np.uint8)
            gray2 = (gray2 * 255).astype(np.uint8)

            flow = cv2.calcOpticalFlowFarneback(
                gray1, gray2,
                None,
                pyr_scale=0.5,
                levels=5,
                winsize=21,
                iterations=3,
                poly_n=7,
                poly_sigma=1.5,
                flags=0
            )
            
            return torch.from_numpy(flow).to(self.device)
        except Exception as e:
            print(f"光流计算错误: {str(e)}")
            return torch.zeros((frame1.shape[1], frame1.shape[2], 2)).to(self.device)
    
    def _get_direction_from_flow(self, flow):
        try:
            mean_x = flow[..., 0].mean().item()
            mean_y = flow[..., 1].mean().item()
            if abs(mean_x) < 0.05 and abs(mean_y) < 0.05:
                return "静止"
            angle = np.arctan2(mean_y, mean_x)
            directions = ["右", "右下", "下", "左下", "左", "左上", "上", "右上"]
            index = int((angle + np.pi) / (2 * np.pi / 8)) % 8
            return directions[index]
        except Exception as e:
            print(f"方向计算错误: {str(e)}")
            return "未知"

############################################################
# 并行视频加载器
############################################################
class ParallelVideoLoader:
    def __init__(self, num_workers=4, batch_size=32):
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.frame_queue = queue.Queue(maxsize=100)
        self.result_queue = queue.Queue()
        self.scene_detector = GPUSceneDetector(batch_size=batch_size)
        self.workers = []
        self.is_running = False
        print(f"并行加载器初始化完成 (workers: {num_workers}, batch_size: {batch_size})")
    
    def start(self):
        self.is_running = True
        self.gpu_thread = threading.Thread(
            target=self._gpu_process_loop,
            name="GPU-Processor",
            daemon=True
        )
        self.gpu_thread.start()
        
        for i in range(self.num_workers):
            worker = threading.Thread(
                target=self._worker_loop,
                name=f"Worker-{i}",
                daemon=True
            )
            worker.start()
            self.workers.append(worker)
        
        print(f"启动了 {self.num_workers} 个工作线程和 1 个GPU处理线程")
    
    def stop(self):
        print("\n正在停止并行处理...")
        self.is_running = False
        self.gpu_thread.join()
        for worker in self.workers:
            worker.join()
        
        while not self.frame_queue.empty():
            try:
                self.frame_queue.get_nowait()
            except queue.Empty:
                break
                
        while not self.result_queue.empty():
            try:
                self.result_queue.get_nowait()
            except queue.Empty:
                break
        
        print("并行处理已停止")
    
    def add_video(self, video_path):
        if not os.path.exists(video_path):
            print(f"错误: 视频文件不存在 - {video_path}")
            return
        self.frame_queue.put(video_path)
        print(f"已添加视频到队列: {os.path.basename(video_path)}")
    
    def _worker_loop(self):
        while self.is_running:
            try:
                video_path = self.frame_queue.get_nowait()
                self._process_video(video_path)
            except queue.Empty:
                time.sleep(0.1)
            except Exception as e:
                print(f"工作线程错误: {str(e)}")
    
    def _process_video(self, video_path):
        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                print(f"无法打开视频: {video_path}")
                return
            frames = []
            frame_count = 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                frame = cv2.resize(frame, CONFIG['FRAME_SIZE'])
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame)
                
                if len(frames) >= self.batch_size:
                    self.result_queue.put(frames)
                    frames = []
                frame_count += 1
                if frame_count % 100 == 0:
                    print(f"已处理 {frame_count} 帧 - {os.path.basename(video_path)}")
            if frames:
                self.result_queue.put(frames)
        except Exception as e:
            print(f"视频处理错误: {str(e)}")
        finally:
            cap.release()
    
    def _gpu_process_loop(self):
        batch_count = 0
        while self.is_running:
            try:
                frames = self.result_queue.get_nowait()
                self.scene_detector.process_batch(frames)
                batch_count += 1
                if batch_count % 10 == 0:
                    print(f"已处理 {batch_count} 批次")
            except queue.Empty:
                time.sleep(0.1)
            except Exception as e:
                print(f"GPU处理错误: {str(e)}")

############################################################
# 增强数据集，使用缓存处理视频序列
############################################################
class EnhancedDataset(Dataset):
    def __init__(self, video_folder, sequence_length=32, frame_size=(256, 256)):
        print("\n初始化数据集...")
        self.video_paths = self._scan_video_folder(video_folder)
        self.sequence_length = sequence_length
        self.frame_size = frame_size
        self.cache_dir = "./tmp/sequences"
        self.sequence_paths = []
        
        os.makedirs(self.cache_dir, exist_ok=True)
        print(f"创建缓存目录: {self.cache_dir}")
        
        self._prepare_data()
        print(f"数据集初始化完成，包含 {len(self.sequence_paths)} 个序列")
    
    def _scan_video_folder(self, folder):
        video_files = []
        video_extensions = ['.mp4', '.avi', '.mov', '.mkv']
        print(f"扫描文件夹: {folder}")
        for ext in video_extensions:
            found_files = glob(os.path.join(folder, f'*{ext}'))
            video_files.extend(found_files)
            if found_files:
                print(f"发现 {len(found_files)} 个{ext}文件")
        return video_files
    
    def _prepare_data(self):
        print("\n开始处理视频数据...")
        for video_idx, video_path in enumerate(self.video_paths):
            print(f"\n处理视频 [{video_idx + 1}/{len(self.video_paths)}]: {os.path.basename(video_path)}")
            try:
                cap = cv2.VideoCapture(video_path)
                if not cap.isOpened():
                    print(f"无法打开视频")
                    continue
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                fps = int(cap.get(cv2.CAP_PROP_FPS))
                print(f"视频信息: {total_frames} 帧, {fps} FPS")
                frames = []
                frame_count = 0
                sequence_count = 0
                
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    frame = cv2.resize(frame, self.frame_size)
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frames.append(frame)
                    frame_count += 1
                    
                    if len(frames) >= self.sequence_length:
                        sequence = np.array(frames)
                        sequence_path = self._save_sequence(sequence, video_idx, sequence_count)
                        self.sequence_paths.append(sequence_path)
                        frames = frames[self.sequence_length//2:]  
                        sequence_count += 1
                        if sequence_count % 100 == 0:
                            frames = frames[-self.sequence_length:]
                            gc.collect()
                    
                    if frame_count % 100 == 0:
                        print(f"\r已处理: {frame_count}/{total_frames} 帧, 创建了 {sequence_count} 个序列", end="")
                
                cap.release()
                print(f"\n视频处理完成: 创建了 {sequence_count} 个序列")
            except Exception as e:
                print(f"处理视频时出错: {str(e)}")
                continue
    
    def _save_sequence(self, sequence, video_idx, sequence_idx):
        filename = f"sequence_{video_idx}_{sequence_idx}.npy"
        filepath = os.path.join(self.cache_dir, filename)
        np.save(filepath, sequence)
        return filepath
    
    def _load_sequence(self, filepath):
        try:
            sequence = np.load(filepath)
            return sequence / 255.0
        except Exception as e:
            print(f"加载序列错误: {str(e)}")
            return np.zeros((self.sequence_length, *self.frame_size, 3))
    
    def __len__(self):
        return len(self.sequence_paths)
    
    def __getitem__(self, idx):
        try:
            sequence = self._load_sequence(self.sequence_paths[idx])
            if random.random() < 0.5:
                sequence = self._augment_sequence(sequence)
            sequence = torch.from_numpy(sequence).float().permute(0, 3, 1, 2)
            frames = sequence[:-1]
            target = sequence[-1:].clone()
            return frames, target
        except Exception as e:
            print(f"获取数据项错误: {str(e)}")
            empty_sequence = torch.zeros(
                (self.sequence_length-1, 3, *self.frame_size),
                dtype=torch.float32
            )
            empty_target = torch.zeros(
                (1, 3, *self.frame_size),
                dtype=torch.float32
            )
            return empty_sequence, empty_target
    
    def _augment_sequence(self, sequence):
        try:
            augmented = sequence.copy()
            if random.random() < 0.5:
                factor = random.uniform(0.8, 1.2)
                augmented = augmented * factor
            if random.random() < 0.5:
                factor = random.uniform(0.8, 1.2)
                mean = augmented.mean()
                augmented = np.clip((augmented - mean) * factor + mean, 0, 1)
            return augmented
        except Exception as e:
            print(f"数据增强错误: {str(e)}")
            return sequence
    
    def cleanup(self):
        try:
            import shutil
            shutil.rmtree(self.cache_dir)
            print(f"清理缓存目录: {self.cache_dir}")
        except Exception as e:
            print(f"清理缓存错误: {str(e)}")

############################################################
# 一些辅助函数
############################################################
def show_sequence(sequence, title="Sequence visualization"):
    if isinstance(sequence, torch.Tensor):
        sequence = sequence.permute(0, 2, 3, 1).cpu().numpy()
    fig, axes = plt.subplots(2, 4, figsize=(15, 8))
    fig.suptitle(title)
    for i, ax in enumerate(axes.flat):
        if i < len(sequence):
            ax.imshow(sequence[i])
            ax.axis('off')
            ax.set_title(f'Frame {i+1}')
    plt.tight_layout()
    plt.show()

def adjust_brightness(frame, factor):
    return np.clip(frame * factor, 0, 255).astype(np.uint8)

def adjust_contrast(frame, factor):
    mean = np.mean(frame, axis=(0, 1), keepdims=True)
    return np.clip((frame - mean) * factor + mean, 0, 255).astype(np.uint8)

def adjust_hue(frame, factor):
    hsv = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)
    hsv[..., 0] = (hsv[..., 0] + factor * 180) % 180
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)

############################################################
# 数据预取器（可选）
############################################################
class DataPrefetcher:
    def __init__(self, loader, device):
        self.loader = iter(loader)
        self.device = device
        self.stream = torch.cuda.Stream()
        self.next_data = None
        self._preload()
    
    def _preload(self):
        try:
            self.next_data = next(self.loader)
        except StopIteration:
            self.next_data = None
            return
        with torch.cuda.stream(self.stream):
            if isinstance(self.next_data, dict):
                for k, v in self.next_data.items():
                    if isinstance(v, torch.Tensor):
                        self.next_data[k] = v.to(self.device, non_blocking=True)
            else:
                self.next_data = [
                    t.to(self.device, non_blocking=True) 
                    for t in self.next_data 
                    if isinstance(t, torch.Tensor)
                ]
    
    def next(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        data = self.next_data
        self._preload()
        return data
    
    def __iter__(self):
        return self
    
    def __next__(self):
        data = self.next()
        if data is None:
            raise StopIteration
        return data

############################################################
# 网络结构：残差块、反卷积块
############################################################
class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1,
                          stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class DeconvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.deconv = nn.ConvTranspose2d(
            in_channels, out_channels,
            kernel_size=4, stride=2,
            padding=1, bias=False
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        return self.relu(self.bn(self.deconv(x)))

############################################################
# 主模型：自监督寻路模型
############################################################
class SelfSupervisedPathfinder(nn.Module):
    def __init__(self, hidden_dim=256):
        super().__init__()
        print("\n初始化神经网络模型...")
        
        self.encoder = nn.Sequential(
            ResBlock(3, 32, stride=2),
            ResBlock(32, 64, stride=2),
            ResBlock(64, 128, stride=2),
            ResBlock(128, 256, stride=1),
            nn.AdaptiveAvgPool2d((8, 8))
        )
        print("特征提取器已创建")
        
        self.motion_encoder = nn.LSTM(
            input_size=256 * 8 * 8,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=0.2
        )
        print("运动编码器已创建")
        
        self.action_predictor = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim // 2, 8)  
        )
        print("动作预测器已创建")
        
        self.frame_predictor = nn.Sequential(
            nn.Linear(hidden_dim * 2, 256 * 8 * 8),
            nn.ReLU(inplace=True),
            nn.Unflatten(1, (256, 8, 8)),
            DeconvBlock(256, 128),
            DeconvBlock(128, 64),
            DeconvBlock(64, 32),
            DeconvBlock(32, 16),
            nn.ConvTranspose2d(16, 3, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid()
        )
        print("帧预测器已创建")
        
        self.apply(self._init_weights)
        print("权重初始化完成")
        self._print_model_info()
    
    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.BatchNorm2d, nn.LayerNorm)):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
    
    def _print_model_info(self):
        print("\n模型结构信息:")
        x = torch.randn(2, 31, 3, 256, 256)
        print("1. 输入尺寸:", x.shape)
        batch_size, seq_len = x.size(0), x.size(1)
        x_reshaped = x.view(batch_size * seq_len, x.size(2), x.size(3), x.size(4))
        features = self.encoder(x_reshaped)
        print("2. 编码器输出:", features.shape)
        features = features.view(batch_size, seq_len, -1)
        lstm_out, _ = self.motion_encoder(features)
        print("3. LSTM输出:", lstm_out.shape)
        final_hidden = lstm_out[:, -1, :]
        actions = self.action_predictor(final_hidden)
        print("4. 动作预测输出:", actions.shape)
        pred_frame = self.frame_predictor(final_hidden)
        print("5. 帧预测输出:", pred_frame.shape, "(期望: [B, 3, 256, 256])")
    
    def forward(self, x):
        batch_size, seq_len = x.size(0), x.size(1)
        x = x.view(batch_size * seq_len, x.size(2), x.size(3), x.size(4))
        features = self.encoder(x)
        features = features.view(batch_size, seq_len, -1)
        lstm_out, _ = self.motion_encoder(features)
        final_hidden = lstm_out[:, -1, :]
        actions = self.action_predictor(final_hidden)
        pred_frame = self.frame_predictor(final_hidden)
        return actions, pred_frame

############################################################
# 模型创建、保存、加载
############################################################
def create_model(pretrained_path=None):
    try:
        print("\n创建模型...")
        model = SelfSupervisedPathfinder()
        if pretrained_path and os.path.exists(pretrained_path):
            print(f"加载预训练权重: {pretrained_path}")
            model.load_state_dict(torch.load(pretrained_path))
        model = model.to(CONFIG['DEVICE'])
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"\n模型信息:")
        print(f"- 总参数量: {total_params:,}")
        print(f"- 可训练参数量: {trainable_params:,}")
        print(f"- 设备: {next(model.parameters()).device}")
        return model
    except Exception as e:
        print(f"创建模型错误: {str(e)}")
        raise e

def save_model(model, path, optimizer=None, epoch=None, loss=None):
    try:
        save_dict = {
            'model_state_dict': model.state_dict(),
        }
        if optimizer is not None:
            save_dict['optimizer_state_dict'] = optimizer.state_dict()
        if epoch is not None:
            save_dict['epoch'] = epoch
        if loss is not None:
            save_dict['loss'] = loss
        torch.save(save_dict, path)
        print(f"模型已保存: {path}")
    except Exception as e:
        print(f"保存模型错误: {str(e)}")

def load_model(path, model=None):
    try:
        if not os.path.exists(path):
            print(f"模型文件不存在: {path}")
            return None, None
        checkpoint = torch.load(path, map_location=CONFIG['DEVICE'])
        if model is None:
            model = SelfSupervisedPathfinder()
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(CONFIG['DEVICE'])
        return model, checkpoint
    except Exception as e:
        print(f"加载模型错误: {str(e)}")
        return None, None

############################################################
# 训练器
############################################################
class Trainer:
    def __init__(self, model, train_loader, val_loader=None, 
                 learning_rate=1e-3, weight_decay=1e-4):
        self.device = CONFIG['DEVICE']
        print(f"\n使用设备: {self.device}")
        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        
        # 优化器设置
        self.optimizer = torch.optim.Adam(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        # 修正：从正确的模块导入学习率调度器
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, 
            patience=5, verbose=True
        )
        
        # 余弦退火学习率
        self.cos_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=50,
            eta_min=1e-6
        )
        
        # 自动混合精度训练
        self.scaler = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())
        
        # 损失权重配置
        self.loss_weights = {
            'frame': 1.0,       # 帧预测损失权重
            'action': 0.1,      # 动作平滑损失权重
            'temporal': 0.05    # 时序一致性损失权重
        }
        
        # 训练状态记录
        self.train_history = {
            'epoch_losses': [],
            'val_losses': [],
            'learning_rates': [],
            'best_loss': float('inf'),
        }
        
        print(f"\n训练器初始化完成:")
        print(f"- 学习率: {learning_rate}")
        print(f"- 权重衰减: {weight_decay}")
        print(f"- 损失权重: {self.loss_weights}")
        if torch.cuda.is_available():
            print(f"- GPU型号: {torch.cuda.get_device_name(0)}")
            print(f"- GPU内存: {torch.cuda.get_device_properties(0).total_memory/1024**3:.1f}GB")

    def train_epoch(self, epoch, is_increment=False):
        """训练一个完整的epoch"""
        self.model.train()
        total_loss = 0
        num_batches = len(self.train_loader)
        progress = tqdm(self.train_loader, desc=f"Epoch {epoch+1}")
        
        for batch_idx, (frames, target) in enumerate(progress):
            try:
                frames = frames.to(self.device)
                target = target.to(self.device)
                loss = self._train_step(frames, target, is_increment)
                total_loss += loss
                
                # 更新进度条信息
                progress.set_postfix({
                    'loss': f"{loss:.4f}",
                    'avg_loss': f"{total_loss/(batch_idx+1):.4f}",
                    'lr': f"{self.optimizer.param_groups[0]['lr']:.6f}"
                })
                
                # 定期执行垃圾回收
                if batch_idx % 100 == 0:
                    torch.cuda.empty_cache()
                    gc.collect()
                    
            except Exception as e:
                print(f"\n批次 {batch_idx} 训练错误: {str(e)}")
                import traceback
                traceback.print_exc()
                continue
                
        epoch_loss = total_loss / num_batches
        self.train_history['epoch_losses'].append(epoch_loss)
        return epoch_loss
    
    def _train_step(self, frames, target, is_increment=False):
        """执行单个训练步骤"""
        self.optimizer.zero_grad()
        
        with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
            # 前向传播
            actions, pred_frame = self.model(frames)
            
            # 计算各项损失
            frame_loss = F.mse_loss(pred_frame, target.squeeze(1))
            action_smooth_loss = self._compute_action_smoothness(actions)
            temporal_loss = self._compute_temporal_consistency(frames, pred_frame)
            
            # 增量训练时调整损失权重
            if is_increment:
                self.loss_weights['frame'] *= 0.8  # 降低帧预测损失权重
                self.loss_weights['temporal'] *= 1.2  # 增加时序一致性权重
            
            # 组合总损失
            loss = (self.loss_weights['frame'] * frame_loss + 
                   self.loss_weights['action'] * action_smooth_loss +
                   self.loss_weights['temporal'] * temporal_loss)
        
        # 反向传播
        self.scaler.scale(loss).backward()
        
        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        
        self.scaler.step(self.optimizer)
        self.scaler.update()
        
        return loss.item()
    
    def _compute_action_smoothness(self, actions):
        """计算动作平滑度损失"""
        return torch.mean(torch.abs(actions))
    
    def _compute_temporal_consistency(self, frames, pred_frame):
        """计算时序一致性损失"""
        last_real_frame = frames[:, -1]
        return F.mse_loss(pred_frame, last_real_frame)
    
    def validate(self):
        """验证模型性能"""
        if self.val_loader is None:
            return None
            
        self.model.eval()
        total_loss = 0
        
        with torch.no_grad():
            for frames, target in tqdm(self.val_loader, desc="验证"):
                try:
                    frames = frames.to(self.device)
                    target = target.to(self.device)
                    actions, pred_frame = self.model(frames)
                    loss = F.mse_loss(pred_frame, target.squeeze(1))
                    total_loss += loss.item()
                except Exception as e:
                    print(f"验证错误: {str(e)}")
                    continue
                    
        val_loss = total_loss / len(self.val_loader)
        self.train_history['val_losses'].append(val_loss)
        
        # 更新学习率
        self.scheduler.step(val_loss)
        self.train_history['learning_rates'].append(
            self.optimizer.param_groups[0]['lr']
        )
        
        # 更新最佳模型
        if val_loss < self.train_history['best_loss']:
            self.train_history['best_loss'] = val_loss
            return True  # 标记需要保存检查点
        return False
    
    def train_incremental(self, video_batches, epochs_per_batch=5, save_dir="./models"):
        """增量训练入口函数"""
        print("\n开始增量训练...")
        for batch_idx, video_batch in enumerate(video_batches):
            print(f"\n处理第 {batch_idx + 1}/{len(video_batches)} 批视频...")
            
            # 创建当前批次的数据集和加载器
            try:
                current_dataset = EnhancedDataset(
                    video_batch,
                    sequence_length=CONFIG['SEQUENCE_LENGTH'],
                    frame_size=CONFIG['FRAME_SIZE']
                )
                
                current_loader = DataLoader(
                    current_dataset,
                    batch_size=CONFIG['BATCH_SIZE'],
                    shuffle=True,
                    num_workers=CONFIG['NUM_WORKERS'],
                    pin_memory=True
                )
                
                # 更新训练器的数据加载器
                self.train_loader = current_loader
                
                # 训练当前批次
                for epoch in range(epochs_per_batch):
                    print(f"\n===> 批次 {batch_idx + 1}, Epoch {epoch + 1}/{epochs_per_batch}")
                    train_loss = self.train_epoch(epoch, is_increment=True)
                    
                    # 执行验证
                    if self.val_loader:
                        is_best = self.validate()
                        if is_best:
                            save_model(
                                self.model,
                                os.path.join(save_dir, f'best_model_batch_{batch_idx}.pth'),
                                self.optimizer,
                                epoch,
                                train_loss
                            )
                    
                    # 保存阶段性检查点
                    if (epoch + 1) % 2 == 0:
                        save_model(
                            self.model,
                            os.path.join(save_dir, f'checkpoint_batch_{batch_idx}_epoch_{epoch+1}.pth'),
                            self.optimizer,
                            epoch,
                            train_loss
                        )
                
                # 清理当前批次的数据集
                current_dataset.cleanup()
                
            except Exception as e:
                print(f"处理批次 {batch_idx} 时发生错误: {str(e)}")
                import traceback
                traceback.print_exc()
                continue
        
        print("\n增量训练完成!")
        
    def plot_training_history(self):
        """绘制训练历史"""
        try:
            import matplotlib.pyplot as plt
            
            plt.figure(figsize=(15, 5))
            
            # 损失曲线
            plt.subplot(1, 2, 1)
            plt.plot(self.train_history['epoch_losses'], label='Training Loss')
            if self.train_history['val_losses']:
                plt.plot(self.train_history['val_losses'], label='Validation Loss')
            plt.title('Loss History')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.legend()
            
            # 学习率曲线
            plt.subplot(1, 2, 2)
            plt.plot(self.train_history['learning_rates'])
            plt.title('Learning Rate History')
            plt.xlabel('Validation Steps')
            plt.ylabel('Learning Rate')
            
            plt.tight_layout()
            plt.savefig('./visualizations/training_history.png')
            plt.close()
            
            print(f"\n训练历史图表已保存至: ./visualizations/training_history.png")
            
        except Exception as e:
            print(f"绘制训练历史出错: {str(e)}")

############################################################
# 训练流程
############################################################
def train_model(model, train_dataset, val_dataset=None, 
                num_epochs=5, batch_size=32, save_dir="./models"):
    print("\n开始训练...")
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    val_loader = None
    if val_dataset is not None:
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=16,
            pin_memory=True
        )
    trainer = Trainer(model, train_loader, val_loader)
    best_loss = float('inf')
    os.makedirs(save_dir, exist_ok=True)
    try:
        for epoch in range(num_epochs):
            train_loss = trainer.train_epoch(epoch)
            val_loss = trainer.validate() if val_loader else None
            print(f"\nEpoch {epoch+1}/{num_epochs} - 训练loss: {train_loss:.4f}")
            if val_loss is not None:
                print(f"              验证loss: {val_loss:.4f}")
            if val_loss is not None:
                trainer.scheduler.step(val_loss)
            if val_loss is not None and val_loss < best_loss:
                best_loss = val_loss
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': trainer.optimizer.state_dict(),
                    'loss': val_loss,
                }, os.path.join(save_dir, 'best_model.pth'))
                print("保存最佳模型")
            if (epoch + 1) % 2 == 0:
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': trainer.optimizer.state_dict(),
                    'loss': train_loss,
                }, os.path.join(save_dir, f'checkpoint_epoch_{epoch+1}.pth'))
                print(f"保存检查点 checkpoint_epoch_{epoch+1}.pth")
    except KeyboardInterrupt:
        print("\n训练被中断 (Ctrl+C)")
    except Exception as e:
        print(f"\n训练错误: {str(e)}")
        import traceback
        traceback.print_exc()
    finally:
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': trainer.optimizer.state_dict(),
            'loss': train_loss,
        }, os.path.join(save_dir, 'final_model.pth'))
        print("保存最终模型\n训练结束")

############################################################
# 推理功能
############################################################
def start_inference_realtime(model, logger=None):
    """实时推理并在透明窗口中可视化显示场景状态"""
    print("\n开始实时推理...\n按 'q' 或 'esc' 退出")
    model.eval()
    
    # 场景状态映射
    scene_states = {
        'MENU': {'name': '菜单界面', 'color': '\033[94m'},  # 蓝色
        'MAP': {'name': '地图界面', 'color': '\033[92m'},   # 绿色
        'DIALOG': {'name': '对话状态', 'color': '\033[93m'}, # 黄色
        'LOADING': {'name': '加载中', 'color': '\033[95m'},  # 紫色
        'COMBAT': {'name': '战斗状态', 'color': '\033[91m'}, # 红色
        'SWIMMING': {'name': '游泳中', 'color': '\033[96m'}, # 青色
        'FLYING': {'name': '飞行中', 'color': '\033[97m'},   # 白色
        'CLIMBING': {'name': '攀爬中', 'color': '\033[90m'}  # 灰色
    }
    
    THRESHOLD = 0.5
    while True:
        if keyboard.is_pressed('q') or keyboard.is_pressed('esc'):
            print("用户退出实时推理")
            break
        
        try:
            # 截图和预处理
            screenshot = pyautogui.screenshot()
            frame = np.array(screenshot)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, CONFIG['FRAME_SIZE'])
            frame = frame.astype(np.float32) / 255.0
            frame_tensor = torch.from_numpy(frame).permute(2,0,1).unsqueeze(0).unsqueeze(1).to(CONFIG['DEVICE'])
            
            # 模型推理
            with torch.no_grad():
                actions, pred_frame = model(frame_tensor)
            
            actions_np = actions.squeeze(0).cpu().numpy()
            
            # 构建可视化日志
            log_lines = []
            log_lines.append("==== 场景状态监测 ====")
            log_lines.append(f"时间: {datetime.now().strftime('%H:%M:%S')}")
            log_lines.append("-" * 30)
            
            # 添加活跃状态
            active_states = []
            for i, (state, config) in enumerate(scene_states.items()):
                prob = actions_np[i]
                if prob > THRESHOLD:
                    active_states.append(f"{config['name']}({prob:.2f})")
            
            if active_states:
                log_lines.append("当前激活状态:")
                for state in active_states:
                    log_lines.append(f"➤ {state}")
            else:
                log_lines.append("⚠ 未检测到明确状态")
            
            log_lines.append("-" * 30)
            log_lines.append("详细状态概率:")
            
            # 添加所有状态概率
            for i, (state, config) in enumerate(scene_states.items()):
                prob = actions_np[i]
                bar_length = int(prob * 20)  # 概率条长度
                bar = "█" * bar_length + "░" * (20 - bar_length)
                log_lines.append(f"{config['name']}: {bar} {prob:.2f}")
            
            # 更新透明窗口日志
            if logger:
                logger.text_box.delete(1.0, tk.END)  # 清除旧内容
                logger.text_box.insert(tk.END, "\n".join(log_lines))
            
            time.sleep(0.5)
            
        except Exception as e:
            error_msg = f"推理错误: {str(e)}"
            print(error_msg)
            if logger:
                logger.add_log(error_msg)
            time.sleep(1)
            continue

def start_inference_video(model, video_path, logger=None):
    print(f"\n开始视频文件推理: {video_path}")
    if not os.path.exists(video_path):
        print("视频文件不存在")
        return
    model.eval()
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("无法打开视频")
        return
    
    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1
        frame = cv2.resize(frame, CONFIG['FRAME_SIZE'])
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = frame.astype(np.float32) / 255.0
        frame_tensor = torch.from_numpy(frame).permute(2,0,1).unsqueeze(0).to(CONFIG['DEVICE'])
        
        with torch.no_grad():
            frame_tensor = frame_tensor.unsqueeze(1)
            actions, pred_frame = model(frame_tensor)
        
        actions_np = actions.squeeze(0).cpu().numpy().round(2)
        msg = f"帧 {frame_count} 动作向量: {actions_np}"
        print(msg)
        if logger:
            logger.add_log(msg)
        
        if frame_count % 30 == 0:
            print(f"已处理 {frame_count} 帧...")
    
    cap.release()
    print("视频推理完成")

############################################################
# 实时截屏训练
############################################################
class RealtimeScreenshotDataset(Dataset):
    def __init__(self, sequence_length=32):
        self.sequence_length = sequence_length
        self.frames = deque(maxlen=sequence_length)
    
    def add_frame(self, frame):
        if isinstance(frame, np.ndarray):
            self.frames.append(frame)
    
    def __len__(self):
        return 9999999
    
    def __getitem__(self, idx):
        while len(self.frames) < self.sequence_length:
            time.sleep(0.01)
        sequence = list(self.frames)
        seq_tensors = []
        for f in sequence:
            f = f.astype(np.float32)/255.0
            t = torch.from_numpy(f).permute(2,0,1)
            seq_tensors.append(t)
        seq_tensor = torch.stack(seq_tensors)
        frames_tensor = seq_tensor[:-1]
        target_tensor = seq_tensor[-1:].clone()
        return frames_tensor, target_tensor

def realtime_screenshot_capture(dataset, stop_event):
    print("开始实时截屏线程...")
    while not stop_event.is_set():
        if keyboard.is_pressed('q') or keyboard.is_pressed('esc'):
            stop_event.set()
            break
        screenshot = pyautogui.screenshot()
        frame = np.array(screenshot)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, CONFIG['FRAME_SIZE'])
        dataset.add_frame(frame)
        time.sleep(0.1)
    print("实时截屏线程结束")

def setup_realtime_training():
    print("\n设置实时截屏训练...")
    realtime_dataset = RealtimeScreenshotDataset(sequence_length=CONFIG['SEQUENCE_LENGTH'])
    train_loader = DataLoader(
        realtime_dataset,
        batch_size=CONFIG['BATCH_SIZE'],
        shuffle=False,
        num_workers=0,
        drop_last=True
    )
    return realtime_dataset, train_loader

############################################################
# 自动操控游戏并自训练
############################################################
def control_game(action_vector):
    forward_prob = action_vector[0]
    jump_prob = action_vector[1]
    # 简单设定阈值进行按键示例
    if forward_prob > 0.5:
        keyboard.press('w')
    else:
        keyboard.release('w')
    if jump_prob > 0.8:
        keyboard.press('space')
        time.sleep(0.1)
        keyboard.release('space')

class SelfTrainGameDataset(Dataset):
    def __init__(self, sequence_length=32):
        self.sequence_length = sequence_length
        self.frames = deque(maxlen=sequence_length)
    
    def add_frame(self, frame):
        self.frames.append(frame)
    
    def __len__(self):
        return 9999999
    
    def __getitem__(self, idx):
        while len(self.frames) < self.sequence_length:
            time.sleep(0.01)
        sequence = list(self.frames)
        seq_tensors = []
        for f in sequence:
            f = f.astype(np.float32)/255.0
            t = torch.from_numpy(f).permute(2,0,1)
            seq_tensors.append(t)
        seq_tensor = torch.stack(seq_tensors)
        frames_tensor = seq_tensor[:-1]
        target_tensor = seq_tensor[-1:].clone()
        return frames_tensor, target_tensor

def game_self_train_loop(model, dataloader, stop_event, logger=None):
    print("开始游戏自训练循环...")
    device = CONFIG['DEVICE']
    model = model.to(device)
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    scaler = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())
    
    while not stop_event.is_set():
        for frames, target in dataloader:
            if stop_event.is_set():
                break
            frames = frames.to(device)
            target = target.to(device)
            
            optimizer.zero_grad()
            with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                actions, pred_frame = model(frames)
                loss_frame = F.mse_loss(pred_frame, target.squeeze(1))
                loss_action = torch.mean(torch.abs(actions))
                loss = loss_frame + 0.1 * loss_action
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            first_action = actions[0].detach().cpu().numpy()
            control_game(first_action)
            
            msg = f"自训练循环, loss={loss.item():.4f}"
            print(msg)
            if logger:
                logger.add_log(msg)
            
            time.sleep(0.1)
    print("结束游戏自训练循环")

def start_self_training(model, logger=None):
    print("\n开始自动操控游戏并自训练...\n按 'q' 或 'esc' 退出")
    self_train_dataset = SelfTrainGameDataset(sequence_length=CONFIG['SEQUENCE_LENGTH'])
    dataloader = DataLoader(
        self_train_dataset,
        batch_size=CONFIG['BATCH_SIZE'],
        shuffle=False,
        num_workers=0,
        drop_last=True
    )
    stop_event = threading.Event()
    
    def capture_loop():
        while not stop_event.is_set():
            if keyboard.is_pressed('q') or keyboard.is_pressed('esc'):
                stop_event.set()
                break
            screenshot = pyautogui.screenshot()
            frame = np.array(screenshot)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, CONFIG['FRAME_SIZE'])
            self_train_dataset.add_frame(frame)
            time.sleep(0.05)
    
    capture_thread = threading.Thread(target=capture_loop, daemon=True)
    capture_thread.start()
    
    try:
        game_self_train_loop(model, dataloader, stop_event, logger=logger)
    except KeyboardInterrupt:
        print("用户手动中断")
    except Exception as e:
        print("自训练过程出错:", e)
    finally:
        stop_event.set()
        capture_thread.join()
        print("自训练结束")

############################################################
# 功能 6：无限向前行走 + 模型避障演示
############################################################
def infinite_forward_and_avoid(model, logger=None):
    """
    功能6：无限向前行走，并利用模型进行避障。
    - 通过跳跃避开低障碍物
    - 双击跳跃（飞行）防止在高落差时摔伤
    - 无法直接避开的过高障碍物则自动攀爬
    - 在左下角透明窗口显示实时日志
    按 'q' 或 'esc' 退出
    """
    print("\n开始功能6：无限向前行走 + 避障...\n按 'q' 或 'esc' 退出")
    device = CONFIG['DEVICE']
    model = model.to(device)
    model.eval()
    
    # 设定一些简单的阈值，模拟障碍检测和动作触发
    jump_threshold = 0.7   # 如果预测需要跳跃动作概率 > 0.7 则跳
    fly_threshold = 0.9    # 如果需要飞行动作概率 > 0.9 则进行双击跳
    climb_threshold = 0.95 # 如果需要攀爬动作概率 > 0.95 则按下攀爬键（假设 shift 之类）
    
    # 持续按下 'w' 实现无限向前行走
    keyboard.press('w')
    
    try:
        while True:
            if keyboard.is_pressed('q') or keyboard.is_pressed('esc'):
                print("用户退出功能6")
                break
            
            # 截图并推理
            screenshot = pyautogui.screenshot()
            frame = np.array(screenshot)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, CONFIG['FRAME_SIZE'])
            frame = frame.astype(np.float32) / 255.0
            frame_tensor = torch.from_numpy(frame).permute(2,0,1).unsqueeze(0).to(device)
            
            with torch.no_grad():
                # seq_len=1 的情况下
                frame_tensor = frame_tensor.unsqueeze(1)
                actions, pred_frame = model(frame_tensor)
            actions_np = actions.squeeze(0).cpu().numpy()
            
            # 模拟动作识别（假设 actions向量：0:forward,1:jump,2:fly,3:climb... 仅示例）
            forward_prob = actions_np[0]
            jump_prob    = actions_np[1]
            fly_prob     = actions_np[2]
            climb_prob   = actions_np[3]
            
            # 控制逻辑
            # 首先确保一直按住 'w'
            if forward_prob < 0.3:
                # 如果 forward_prob 不高，也一直保持前进
                keyboard.press('w')
            
            # 低障碍物 => 跳跃
            if jump_prob > jump_threshold:
                keyboard.press('space')
                time.sleep(0.1)
                keyboard.release('space')
            
            # 高落差 => 双击跳跃进入飞行
            if fly_prob > fly_threshold:
                # 双击跳
                keyboard.press('space')
                time.sleep(0.08)
                keyboard.release('space')
                time.sleep(0.1)
                keyboard.press('space')
                time.sleep(0.08)
                keyboard.release('space')
            
            # 无法避开的过高障碍物 => 攀爬
            if climb_prob > climb_threshold:
                # 假设按下某个键进行攀爬
                keyboard.press('shift')
                time.sleep(0.5)
                keyboard.release('shift')
            
            msg = (f"动作预测 => forward={forward_prob:.2f}, jump={jump_prob:.2f}, "
                   f"fly={fly_prob:.2f}, climb={climb_prob:.2f}")
            print(msg)
            if logger:
                logger.add_log(msg)
            
            time.sleep(0.3)
    finally:
        # 释放按键
        keyboard.release('w')
        print("功能6结束")

############################################################
# 主流程：交互式选择运行模式
############################################################
def choose_mode():
    print("\n请选择运行模式:")
    print("1. 视频文件训练")
    print("2. 实时截屏训练")
    print("3. 实时推理")
    print("4. 视频文件推理")
    print("5. 自动操控游戏并自训练")
    print("6. 无限往前走 + 避障（跳跃/飞行/攀爬）演示")
    while True:
        try:
            choice = input("\n请输入选择 (1-6): ").strip()
            if choice in ['1','2','3','4','5','6']:
                return int(choice)
            print("无效选择，请重试")
        except Exception as e:
            print(f"输入错误: {str(e)}")
            print("请重新选择")

def setup_video_training():
    print("\n设置视频训练...")
    video_dir = "./videos"
    if not os.path.exists(video_dir):
        print(f"创建视频文件夹: {video_dir}")
        os.makedirs(video_dir)
    video_files = glob(os.path.join(video_dir, "*.*"))
    if not video_files:
        print(f"\n错误: 未找到视频文件，请将训练视频放入 {video_dir} 文件夹")
        return None
    print(f"\n找到 {len(video_files)} 个视频文件:")
    for file in video_files:
        print(f"- {os.path.basename(file)}")
    try:
        dataset = EnhancedDataset(
            video_folder=video_dir,
            sequence_length=CONFIG['SEQUENCE_LENGTH'],
            frame_size=CONFIG['FRAME_SIZE']
        )
        return dataset
    except Exception as e:
        print(f"\n创建数据集错误: {str(e)}")
        return None

############################################################
# 主函数
############################################################
def main():
    try:
        mode = choose_mode()
        if mode in [2, 5, 6]:
            CONFIG['GPU_MEMORY_FRACTION'] = 0.4
        
        setup_environment()
        
        # 检查已有模型
        model_path = "./models/final_model.pth"
        if os.path.exists(model_path):
            print(f"\n发现已有模型: {model_path}")
            model, checkpoint = load_model(model_path)
            if model is None:
                print("加载已有模型失败，创建新模型...")
                model = create_model()
            else:
                print("成功加载已有模型，将使用增量训练")
        else:
            print("\n未发现已有模型，创建新模型...")
            model = create_model()
        
        # 初始化日志记录器
        logger = OverlayLogger()
        
        if mode == 1:
            # 视频文件训练模式
            video_dir = "./videos"
            if not os.path.exists(video_dir):
                print(f"创建视频文件夹: {video_dir}")
                os.makedirs(video_dir)
            
            video_files = glob(os.path.join(video_dir, "*.*"))
            if not video_files:
                print(f"\n错误: 未找到视频文件，请将训练视频放入 {video_dir} 文件夹")
                return
            
            try:
                dataset = EnhancedDataset(
                    video_folder=video_dir,
                    sequence_length=CONFIG['SEQUENCE_LENGTH'],
                    frame_size=CONFIG['FRAME_SIZE']
                )
                
                train_loader = DataLoader(
                    dataset,
                    batch_size=CONFIG['BATCH_SIZE'],
                    shuffle=True,
                    num_workers=CONFIG['NUM_WORKERS'],
                    pin_memory=True
                )
                
                # 创建训练器
                trainer = Trainer(model, train_loader)
                
                # 执行训练
                print("\n开始训练...")
                for epoch in range(5):  # 5个epoch
                    trainer.train_epoch(epoch)
                    trainer.validate()
                
                # 绘制训练历史
                trainer.plot_training_history()
                
                # 保存最终模型
                save_model(
                    model,
                    "./models/final_model.pth",
                    trainer.optimizer,
                    epoch=5,
                    loss=trainer.train_history['epoch_losses'][-1]
                )
                
                print("\n训练完成!")
                
            except Exception as e:
                print(f"\n训练过程出错: {str(e)}")
                import traceback
                traceback.print_exc()
            finally:
                # 清理数据集
                if 'dataset' in locals():
                    dataset.cleanup()
        
        elif mode == 2:
            # 实时截屏训练
            logger.start()
            realtime_dataset, train_loader = setup_realtime_training()
            stop_event = threading.Event()
            capture_thread = threading.Thread(
                target=realtime_screenshot_capture, 
                args=(realtime_dataset, stop_event),
                daemon=True
            )
            capture_thread.start()
            
            trainer = Trainer(model, train_loader)
            try:
                for epoch in range(3):
                    if stop_event.is_set():
                        break
                    print(f"\n===> Realtime Training Epoch {epoch+1}")
                    trainer.train_epoch(epoch)
            except KeyboardInterrupt:
                print("用户中断实时训练")
            finally:
                stop_event.set()
                capture_thread.join()
                torch.save(model.state_dict(), "./models/realtime_screenshot_model.pth")
                print("已保存实时截屏训练模型")
            logger.stop()
        
        elif mode == 3:
            # 实时推理
            logger.start()
            start_inference_realtime(model, logger=logger)
            logger.stop()
        
        elif mode == 4:
            # 视频文件推理
            video_files = glob("./videos/*.*")
            if not video_files:
                print("\n错误: 未找到视频文件")
                return
            print("\n选择视频文件:")
            for i, file in enumerate(video_files):
                print(f"{i+1}. {os.path.basename(file)}")
            while True:
                try:
                    choice = int(input(f"\n请选择视频文件 (1-{len(video_files)}): "))
                    if 1 <= choice <= len(video_files):
                        video_path = video_files[choice-1]
                        break
                    print("无效选择，请重试")
                except ValueError:
                    print("请输入数字")
            logger.start()
            start_inference_video(model, video_path, logger=logger)
            logger.stop()
        
        elif mode == 5:
            # 自动操控游戏并自训练
            logger.start()
            start_self_training(model, logger=logger)
            logger.stop()
        
        else:
            # 功能6：无限向前走 + 避障
            logger.start()
            infinite_forward_and_avoid(model, logger=logger)
            logger.stop()
    
    except KeyboardInterrupt:
        print("\n程序被中断 (Ctrl+C)")
    except Exception as e:
        print(f"\n程序错误: {str(e)}")
        import traceback
        traceback.print_exc()
    finally:
        cleanup()
        print("\n程序结束")

if __name__ == "__main__":
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    main()
