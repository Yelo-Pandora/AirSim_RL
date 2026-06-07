import tkinter as tk
from tkinter import ttk, messagebox
import os
import sys
import json
import threading
import numpy as np

# Add project root to path
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

class NavGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("AirSim A* + TD3 导航管线")
        self.root.geometry("1050x560")
        
        self.pipeline = None
        self.waypoints = []
        self.point_select_mode = tk.StringVar(value="start")
        self.map_photo = None
        self.map_canvas = None
        self.map_image_item = None
        self.map_info_var = tk.StringVar(value="")
        self.map_origin_row = None
        self.map_origin_col = None
        self.map_mpp_x = None
        self.map_mpp_y = None
        self.start_marker_id = None
        self.end_marker_id = None
        
        self.create_widgets()

    def _get_pipeline(self):
        if self.pipeline is not None:
            return self.pipeline

        try:
            from nav_pipeline import NavPipeline
        except Exception as e:
            messagebox.showerror(
                "依赖缺失",
                f"无法导入导航管线 nav_pipeline：{e}\n\n"
                "地图选点仍可使用；如需路径规划/执行，请切换到含 AirSim 依赖的 Python 环境运行。",
            )
            return None

        try:
            self.pipeline = NavPipeline()
        except Exception as e:
            messagebox.showerror("初始化失败", f"NavPipeline 初始化失败：{e}")
            return None
        return self.pipeline
        
    def create_widgets(self):
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill="both", expand=True)

        main_frame.columnconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=0)
        main_frame.rowconfigure(0, weight=1)

        left_frame = ttk.Frame(main_frame)
        left_frame.grid(row=0, column=0, sticky="nsew")
        right_frame = ttk.Frame(main_frame)
        right_frame.grid(row=0, column=1, sticky="nsew", padx=(10, 10), pady=(10, 10))

        title_label = ttk.Label(left_frame, text="AirSim 智能导航管线", font=("Arial", 16, "bold"))
        title_label.pack(pady=10)

        env_frame = ttk.LabelFrame(left_frame, text="环境控制")
        env_frame.pack(fill="x", padx=10, pady=5)

        self.launch_btn = ttk.Button(env_frame, text="启动 AirSim", command=self.launch_airsim)
        self.launch_btn.pack(side="left", padx=10, pady=5)

        self.init_btn = ttk.Button(env_frame, text="初始化模型", command=self.init_model)
        self.init_btn.pack(side="left", padx=10, pady=5)

        pos_frame = ttk.LabelFrame(left_frame, text="导航设置 (X, Y, Z)")
        pos_frame.pack(fill="x", padx=10, pady=5)

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

        action_frame = ttk.Frame(left_frame)
        action_frame.pack(fill="x", padx=10, pady=10)

        self.plan_btn = ttk.Button(action_frame, text="A* 路径规划", command=self.plan_path)
        self.plan_btn.pack(side="left", expand=True, padx=5)

        self.nav_btn = ttk.Button(action_frame, text="执行 TD3 导航", command=self.run_navigation, state="disabled")
        self.nav_btn.pack(side="left", expand=True, padx=5)

        status_frame = ttk.LabelFrame(left_frame, text="运行状态")
        status_frame.pack(fill="both", expand=True, padx=10, pady=(0, 10))

        self.status_text = tk.Text(status_frame, height=10, state="disabled")
        self.status_text.pack(fill="both", expand=True, padx=5, pady=5)

        self._create_map_panel(right_frame)

    def _create_map_panel(self, parent):
        map_frame = ttk.LabelFrame(parent, text="地图选点")
        map_frame.pack(fill="both", expand=True)

        option_frame = ttk.Frame(map_frame)
        option_frame.pack(fill="x", padx=10, pady=(8, 4))

        ttk.Radiobutton(option_frame, text="设置起点", value="start", variable=self.point_select_mode).pack(
            side="left", padx=(0, 10)
        )
        ttk.Radiobutton(option_frame, text="设置终点", value="end", variable=self.point_select_mode).pack(side="left")

        info_label = ttk.Label(map_frame, textvariable=self.map_info_var)
        info_label.pack(fill="x", padx=10, pady=(0, 6))

        png_path = os.path.join(
            REPO_ROOT,
            "Network",
            "Model6",
            "topdown_depth",
            "topdown_depth_20260601_165808_occupancy_h10_infl2.png",
        )
        metadata_path = os.path.join(
            REPO_ROOT,
            "Network",
            "Model6",
            "topdown_depth",
            "topdown_depth_20260601_165808_occupancy_h10_infl2.json",
        )

        try:
            with open(metadata_path, "r", encoding="utf-8") as f:
                metadata = json.load(f)
            frame = metadata.get("coordinate_frame", {})
            origin = frame.get("origin_pixel", {})
            meters = metadata.get("meters_per_pixel", {})
            self.map_origin_row = float(origin["row"])
            self.map_origin_col = float(origin["col"])
            self.map_mpp_x = float(meters["x"])
            self.map_mpp_y = float(meters["y"])
        except Exception as e:
            self.map_info_var.set(f"元数据加载失败: {e}")
            ttk.Label(map_frame, text="无法加载地图元数据").pack(fill="both", expand=True, padx=10, pady=10)
            return

        try:
            self.map_photo = tk.PhotoImage(file=png_path)
        except Exception as e:
            self.map_info_var.set(f"图片加载失败: {e}")
            ttk.Label(map_frame, text="无法加载地图图片").pack(fill="both", expand=True, padx=10, pady=10)
            return

        self.map_canvas = tk.Canvas(
            map_frame,
            width=self.map_photo.width(),
            height=self.map_photo.height(),
            highlightthickness=0,
        )
        self.map_canvas.pack(fill="both", expand=False, padx=10, pady=(0, 10))
        self.map_image_item = self.map_canvas.create_image(0, 0, anchor="nw", image=self.map_photo)
        self.map_canvas.bind("<Button-1>", self._on_map_click)

    def _set_entry_value(self, entry, value):
        entry.delete(0, "end")
        entry.insert(0, f"{value:.2f}")

    def _update_marker(self, row, col, which):
        if self.map_canvas is None:
            return
        radius = 5
        x0 = col - radius
        y0 = row - radius
        x1 = col + radius
        y1 = row + radius
        color = "#00aa00" if which == "start" else "#0066ff"
        item_id = self.start_marker_id if which == "start" else self.end_marker_id
        if item_id is None:
            item_id = self.map_canvas.create_oval(x0, y0, x1, y1, outline=color, width=2)
            if which == "start":
                self.start_marker_id = item_id
            else:
                self.end_marker_id = item_id
        else:
            self.map_canvas.coords(item_id, x0, y0, x1, y1)

    def _on_map_click(self, event):
        if self.map_photo is None or self.map_canvas is None:
            return

        col = int(event.x)
        row = int(event.y)

        if not (0 <= col < self.map_photo.width() and 0 <= row < self.map_photo.height()):
            return

        x = (self.map_origin_row - float(row)) * self.map_mpp_x
        y = (float(col) - self.map_origin_col) * self.map_mpp_y

        mode = self.point_select_mode.get()
        if mode == "start":
            self._set_entry_value(self.start_x, x)
            self._set_entry_value(self.start_y, y)
        else:
            self._set_entry_value(self.end_x, x)
            self._set_entry_value(self.end_y, y)

        self._update_marker(row, col, mode)
        self.map_info_var.set(f"像素(row={row}, col={col}) -> 坐标(x={x:.2f}, y={y:.2f})")
        self.log(f"[地图选点] {self.map_info_var.get()}")
        
    def log(self, message):
        self.status_text.config(state="normal")
        self.status_text.insert("end", f"{message}\n")
        self.status_text.see("end")
        self.status_text.config(state="disabled")
        
    def launch_airsim(self):
        pipeline = self._get_pipeline()
        if pipeline is None:
            return

        success, msg = pipeline.launch_airsim()
        self.log(msg)
        if success:
            messagebox.showinfo("成功", "AirSim 已启动")
            
    def init_model(self):
        pipeline = self._get_pipeline()
        if pipeline is None:
            return

        def task():
            self.log("正在初始化模型...")
            success, msg = pipeline.init_navigation()
            self.log(msg)
            if success:
                self.root.after(0, lambda: messagebox.showinfo("成功", "模型加载完成"))
        
        threading.Thread(target=task).start()
        
    def plan_path(self):
        pipeline = self._get_pipeline()
        if pipeline is None:
            return

        try:
            start = np.array([float(self.start_x.get()), float(self.start_y.get()), float(self.start_z.get())])
            end = np.array([float(self.end_x.get()), float(self.end_y.get()), float(self.end_z.get())])
            
            self.log(f"开始 A* 规划: {start} -> {end}")
            
            def task():
                self.waypoints, msg = pipeline.plan_path(start, end)
                self.log(msg)
                if self.waypoints:
                    self.log(f"生成了 {len(self.waypoints)} 个局部目标点")
                    self.root.after(0, lambda: self.nav_btn.config(state="normal"))
            
            threading.Thread(target=task).start()
            
        except ValueError:
            messagebox.showerror("错误", "请输入有效的坐标数值")

    def run_navigation(self):
        pipeline = self._get_pipeline()
        if pipeline is None:
            return

        if not self.waypoints:
            messagebox.showwarning("警告", "请先进行路径规划")
            return
            
        def task():
            self.log("开始 TD3 导航...")
            self.nav_btn.config(state="disabled")
            success, msg = pipeline.run_navigation(self.waypoints, status_callback=self.log)
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
