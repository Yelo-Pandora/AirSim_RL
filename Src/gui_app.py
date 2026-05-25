import tkinter as tk
from tkinter import ttk, messagebox
import os
import sys
import threading
import numpy as np

# Add project root to path
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from nav_pipeline import NavPipeline

class NavGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("AirSim A* + TD3 导航管线")
        self.root.geometry("600x500")
        
        self.pipeline = NavPipeline()
        self.waypoints = []
        
        self.create_widgets()
        
    def create_widgets(self):
        # Title
        title_label = ttk.Label(self.root, text="AirSim 智能导航管线", font=("Arial", 16, "bold"))
        title_label.pack(pady=10)
        
        # Environment Control
        env_frame = ttk.LabelFrame(self.root, text="环境控制")
        env_frame.pack(fill="x", padx=10, pady=5)
        
        self.launch_btn = ttk.Button(env_frame, text="启动 AirSim", command=self.launch_airsim)
        self.launch_btn.pack(side="left", padx=10, pady=5)
        
        self.init_btn = ttk.Button(env_frame, text="初始化模型", command=self.init_model)
        self.init_btn.pack(side="left", padx=10, pady=5)
        
        # Position Input
        pos_frame = ttk.LabelFrame(self.root, text="导航设置 (X, Y, Z)")
        pos_frame.pack(fill="x", padx=10, pady=5)
        
        # Start Pos
        start_label = ttk.Label(pos_frame, text="起点:")
        start_label.grid(row=0, column=0, padx=5, pady=5)
        
        self.start_x = ttk.Entry(pos_frame, width=8)
        self.start_x.insert(0, "0")
        self.start_x.grid(row=0, column=1, padx=2)
        
        self.start_y = ttk.Entry(pos_frame, width=8)
        self.start_y.insert(0, "0")
        self.start_y.grid(row=0, column=2, padx=2)
        
        self.start_z = ttk.Entry(pos_frame, width=8)
        self.start_z.insert(0, "-2")
        self.start_z.grid(row=0, column=3, padx=2)
        
        # End Pos
        end_label = ttk.Label(pos_frame, text="终点:")
        end_label.grid(row=1, column=0, padx=5, pady=5)
        
        self.end_x = ttk.Entry(pos_frame, width=8)
        self.end_x.insert(0, "40")
        self.end_x.grid(row=1, column=1, padx=2)
        
        self.end_y = ttk.Entry(pos_frame, width=8)
        self.end_y.insert(0, "40")
        self.end_y.grid(row=1, column=2, padx=2)
        
        self.end_z = ttk.Entry(pos_frame, width=8)
        self.end_z.insert(0, "-2")
        self.end_z.grid(row=1, column=3, padx=2)
        
        # Actions
        action_frame = ttk.Frame(self.root)
        action_frame.pack(fill="x", padx=10, pady=10)
        
        self.plan_btn = ttk.Button(action_frame, text="A* 路径规划", command=self.plan_path)
        self.plan_btn.pack(side="left", expand=True, padx=5)
        
        self.nav_btn = ttk.Button(action_frame, text="执行 TD3 导航", command=self.run_navigation, state="disabled")
        self.nav_btn.pack(side="left", expand=True, padx=5)
        
        # Status
        status_frame = ttk.LabelFrame(self.root, text="运行状态")
        status_frame.pack(fill="both", expand=True, padx=10, pady=5)
        
        self.status_text = tk.Text(status_frame, height=10, state="disabled")
        self.status_text.pack(fill="both", expand=True, padx=5, pady=5)
        
    def log(self, message):
        self.status_text.config(state="normal")
        self.status_text.insert("end", f"{message}\n")
        self.status_text.see("end")
        self.status_text.config(state="disabled")
        
    def launch_airsim(self):
        success, msg = self.pipeline.launch_airsim()
        self.log(msg)
        if success:
            messagebox.showinfo("成功", "AirSim 已启动")
            
    def init_model(self):
        def task():
            self.log("正在初始化模型...")
            success, msg = self.pipeline.init_navigation()
            self.log(msg)
            if success:
                self.root.after(0, lambda: messagebox.showinfo("成功", "模型加载完成"))
        
        threading.Thread(target=task).start()
        
    def plan_path(self):
        try:
            start = np.array([float(self.start_x.get()), float(self.start_y.get()), float(self.start_z.get())])
            end = np.array([float(self.end_x.get()), float(self.end_y.get()), float(self.end_z.get())])
            
            self.log(f"开始 A* 规划: {start} -> {end}")
            
            def task():
                self.waypoints, msg = self.pipeline.plan_path(start, end)
                self.log(msg)
                if self.waypoints:
                    self.log(f"生成了 {len(self.waypoints)} 个局部目标点")
                    self.root.after(0, lambda: self.nav_btn.config(state="normal"))
            
            threading.Thread(target=task).start()
            
        except ValueError:
            messagebox.showerror("错误", "请输入有效的坐标数值")

    def run_navigation(self):
        if not self.waypoints:
            messagebox.showwarning("警告", "请先进行路径规划")
            return
            
        def task():
            self.log("开始 TD3 导航...")
            self.nav_btn.config(state="disabled")
            success, msg = self.pipeline.run_navigation(self.waypoints, status_callback=self.log)
            self.log(msg)
            if success:
                self.root.after(0, lambda: messagebox.showinfo("完成", "任务成功完成！"))
            else:
                self.root.after(0, lambda: messagebox.showerror("失败", msg))
            self.root.after(0, lambda: self.nav_btn.config(state="normal"))
            
        threading.Thread(target=task).start()

if __name__ == "__main__":
    root = tk.Tk()
    app = NavGUI(root)
    root.mainloop()
