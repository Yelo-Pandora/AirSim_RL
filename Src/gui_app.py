import json
import os
import sys
import threading
import tkinter as tk
from tkinter import messagebox, ttk

import numpy as np


REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
ASTAR_DIR = os.path.join(REPO_ROOT, "Network", "Astar_planner")

for path in (REPO_ROOT, ASTAR_DIR):
    if path not in sys.path:
        sys.path.insert(0, path)

try:
    import config as model6_config
except Exception:
    model6_config = None


UE_ORIGIN_X = 1910.0
UE_ORIGIN_Y = -458.0
UE_ORIGIN_Z = 100.0
DEMO_CRUISE_Z = -20.0


def ue_world_to_airsim_ned(x, y, z):
    """Convert dataset/UE-world centimeters to AirSim local NED meters."""
    rel_x = (float(x) - UE_ORIGIN_X) / 100.0
    rel_y = (float(y) - UE_ORIGIN_Y) / 100.0
    rel_z = -((float(z) - UE_ORIGIN_Z) / 100.0)
    return np.array([rel_x, rel_y, rel_z], dtype=np.float32)


def normalize_demo_point(values):
    """Auto-detect whether values are UE-world or already AirSim-local."""
    point = np.array([float(value) for value in values], dtype=np.float32)
    if max(abs(float(value)) for value in point) > 500.0:
        return ue_world_to_airsim_ned(*point), "UE world -> AirSim local"
    return point, "AirSim local"


class NavGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("AirSim 分层导航演示")
        self.root.geometry("1180x720")

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
        self.path_item_ids = []

        self.create_widgets()

    def _get_pipeline(self):
        if self.pipeline is not None:
            return self.pipeline

        try:
            from nav_pipeline import NavPipeline
        except Exception as exc:
            messagebox.showerror(
                "依赖缺失",
                f"无法导入导航管线 nav_pipeline：{exc}\n\n"
                "地图选点仍可使用；如需规划/执行，请切换到含 AirSim 依赖的 Python 环境。",
            )
            return None

        try:
            self.pipeline = NavPipeline(planner_mode="occupancy")
        except Exception as exc:
            messagebox.showerror("初始化失败", f"NavPipeline 初始化失败：{exc}")
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

        title_label = ttk.Label(left_frame, text="AirSim A* + TD3 分层导航演示", font=("Microsoft YaHei UI", 16, "bold"))
        title_label.pack(pady=10)

        self._create_env_panel(left_frame)
        self._create_position_panel(left_frame)
        self._create_action_panel(left_frame)
        self._create_status_panel(left_frame)
        self._create_map_panel(right_frame)

    def _create_env_panel(self, parent):
        env_frame = ttk.LabelFrame(parent, text="环境控制")
        env_frame.pack(fill="x", padx=10, pady=5)

        self.launch_btn = ttk.Button(env_frame, text="启动 AirSim", command=self.launch_airsim)
        self.launch_btn.pack(side="left", padx=10, pady=5)

        self.init_btn = ttk.Button(env_frame, text="初始化模型/连接", command=self.init_model)
        self.init_btn.pack(side="left", padx=10, pady=5)

        planner_text = "当前上层规划器: occupancy A*"
        if model6_config is not None:
            shield = "开" if bool(getattr(model6_config, "SAFETY_SHIELD_ENABLED", False)) else "关"
            spacing = getattr(model6_config, "LOCAL_TARGET_SPACING", "?")
            min_spacing = getattr(model6_config, "LOCAL_TARGET_MIN_SPACING", "?")
            planner_text = f"occupancy A* | 局部目标间隔 {spacing}/{min_spacing}m | Safety Shield: {shield}"
        ttk.Label(env_frame, text=planner_text).pack(side="left", padx=10)

    def _create_position_panel(self, parent):
        pos_frame = ttk.LabelFrame(parent, text="演示坐标 (AirSim local NED；若输入 UE 世界坐标会自动转换)")
        pos_frame.pack(fill="x", padx=10, pady=5)

        ttk.Label(pos_frame, text="起点:").grid(row=0, column=0, padx=5, pady=5)
        self.start_x = ttk.Entry(pos_frame, width=9)
        self.start_x.insert(0, "0")
        self.start_x.grid(row=0, column=1, padx=2)
        self.start_y = ttk.Entry(pos_frame, width=9)
        self.start_y.insert(0, "0")
        self.start_y.grid(row=0, column=2, padx=2)
        self.start_z = ttk.Entry(pos_frame, width=9)
        self.start_z.insert(0, f"{DEMO_CRUISE_Z:.0f}")
        self.start_z.grid(row=0, column=3, padx=2)

        ttk.Label(pos_frame, text="终点:").grid(row=1, column=0, padx=5, pady=5)
        self.end_x = ttk.Entry(pos_frame, width=9)
        self.end_x.insert(0, "40")
        self.end_x.grid(row=1, column=1, padx=2)
        self.end_y = ttk.Entry(pos_frame, width=9)
        self.end_y.insert(0, "40")
        self.end_y.grid(row=1, column=2, padx=2)
        self.end_z = ttk.Entry(pos_frame, width=9)
        self.end_z.insert(0, f"{DEMO_CRUISE_Z:.0f}")
        self.end_z.grid(row=1, column=3, padx=2)

        ttk.Label(pos_frame, text="提示: 地图点击只改 X/Y，Z 默认保持 -20m 演示巡航高度。").grid(
            row=2, column=0, columnspan=4, sticky="w", padx=5, pady=(0, 5)
        )

    def _create_action_panel(self, parent):
        action_frame = ttk.Frame(parent)
        action_frame.pack(fill="x", padx=10, pady=10)

        self.plan_btn = ttk.Button(action_frame, text="A* 路径规划", command=self.plan_path)
        self.plan_btn.pack(side="left", expand=True, padx=5)

        self.clear_btn = ttk.Button(action_frame, text="清除地图路径", command=self.clear_path_overlay)
        self.clear_btn.pack(side="left", expand=True, padx=5)

        self.nav_btn = ttk.Button(action_frame, text="执行 TD3 导航", command=self.run_navigation, state="disabled")
        self.nav_btn.pack(side="left", expand=True, padx=5)

    def _create_status_panel(self, parent):
        status_frame = ttk.LabelFrame(parent, text="运行状态")
        status_frame.pack(fill="both", expand=True, padx=10, pady=(0, 10))

        self.status_text = tk.Text(status_frame, height=12, state="disabled", wrap="word")
        self.status_text.pack(fill="both", expand=True, padx=5, pady=5)

    def _resolve_map_paths(self):
        default_metadata = os.path.join(
            ASTAR_DIR,
            "topdown_depth",
            "topdown_depth_20260601_165808_occupancy_h10_infl2.json",
        )
        metadata_path = getattr(model6_config, "OCCUPANCY_TOPDOWN_METADATA", default_metadata)
        metadata_path = os.path.abspath(metadata_path)

        with open(metadata_path, "r", encoding="utf-8") as file:
            metadata = json.load(file)

        png_candidates = []
        png_from_metadata = metadata.get("files", {}).get("occupancy_png")
        if png_from_metadata:
            if os.path.isabs(png_from_metadata):
                png_candidates.append(png_from_metadata)
            else:
                png_candidates.append(os.path.join(REPO_ROOT, png_from_metadata))
                png_candidates.append(os.path.join(os.path.dirname(metadata_path), os.path.basename(png_from_metadata)))
        png_candidates.append(os.path.splitext(metadata_path)[0] + ".png")

        png_path = next((path for path in png_candidates if os.path.exists(path)), png_candidates[-1])
        return metadata_path, png_path, metadata

    def _create_map_panel(self, parent):
        map_frame = ttk.LabelFrame(parent, text="地图选点与 A* 路径预览")
        map_frame.pack(fill="both", expand=True)

        option_frame = ttk.Frame(map_frame)
        option_frame.pack(fill="x", padx=10, pady=(8, 4))

        ttk.Radiobutton(option_frame, text="设置起点", value="start", variable=self.point_select_mode).pack(
            side="left", padx=(0, 10)
        )
        ttk.Radiobutton(option_frame, text="设置终点", value="end", variable=self.point_select_mode).pack(side="left")

        info_label = ttk.Label(map_frame, textvariable=self.map_info_var)
        info_label.pack(fill="x", padx=10, pady=(0, 6))

        try:
            metadata_path, png_path, metadata = self._resolve_map_paths()
            frame = metadata.get("coordinate_frame", {})
            coverage = metadata.get("coverage", {})
            meters = metadata.get("meters_per_pixel", {})
            self.map_mpp_x = float(meters["x"])
            self.map_mpp_y = float(meters["y"])

            if frame.get("origin") == "vehicle_spawn":
                origin = frame["origin_pixel"]
                self.map_origin_row = float(origin["row"])
                self.map_origin_col = float(origin["col"])
            else:
                self.map_origin_row = float(coverage["x_max"]) / self.map_mpp_x
                self.map_origin_col = -float(coverage["y_min"]) / self.map_mpp_y

            self.map_photo = tk.PhotoImage(file=png_path)
            self.map_info_var.set(f"地图: {os.path.basename(metadata_path)}")
        except Exception as exc:
            self.map_info_var.set(f"地图加载失败: {exc}")
            ttk.Label(map_frame, text="无法加载 topdown occupancy 地图。").pack(fill="both", expand=True, padx=10, pady=10)
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

    def _append_log(self, message):
        self.status_text.config(state="normal")
        self.status_text.insert("end", f"{message}\n")
        self.status_text.see("end")
        self.status_text.config(state="disabled")

    def log(self, message):
        if threading.current_thread() is threading.main_thread():
            self._append_log(str(message))
        else:
            self.root.after(0, self._append_log, str(message))

    def _set_entry_value(self, entry, value):
        entry.delete(0, "end")
        entry.insert(0, f"{value:.2f}")

    def _canvas_to_world_xy(self, row, col):
        x = (self.map_origin_row - float(row)) * self.map_mpp_x
        y = (float(col) - self.map_origin_col) * self.map_mpp_y
        return x, y

    def _world_to_canvas_xy(self, point):
        row = self.map_origin_row - float(point[0]) / self.map_mpp_x
        col = self.map_origin_col + float(point[1]) / self.map_mpp_y
        return col, row

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

        x, y = self._canvas_to_world_xy(row, col)
        mode = self.point_select_mode.get()
        if mode == "start":
            self._set_entry_value(self.start_x, x)
            self._set_entry_value(self.start_y, y)
        else:
            self._set_entry_value(self.end_x, x)
            self._set_entry_value(self.end_y, y)

        self._update_marker(row, col, mode)
        self.map_info_var.set(f"像素(row={row}, col={col}) -> AirSim local (x={x:.2f}, y={y:.2f})")
        self.log(f"[地图选点] {self.map_info_var.get()}")

    def clear_path_overlay(self):
        if self.map_canvas is None:
            return
        for item_id in self.path_item_ids:
            self.map_canvas.delete(item_id)
        self.path_item_ids = []
        self.waypoints = []
        self.nav_btn.config(state="disabled")
        self.log("已清除地图路径。")

    def draw_waypoints(self, waypoints):
        if self.map_canvas is None or not waypoints:
            return
        for item_id in self.path_item_ids:
            self.map_canvas.delete(item_id)
        self.path_item_ids = []

        canvas_points = [self._world_to_canvas_xy(point) for point in waypoints]
        if len(canvas_points) >= 2:
            coords = []
            for col, row in canvas_points:
                coords.extend([col, row])
            self.path_item_ids.append(
                self.map_canvas.create_line(*coords, fill="#00b7ff", width=2, dash=(6, 3))
            )

        for index, (col, row) in enumerate(canvas_points):
            radius = 4 if index not in (0, len(canvas_points) - 1) else 6
            if index == 0:
                color = "#20c933"
            elif index == len(canvas_points) - 1:
                color = "#2f6fff"
            else:
                color = "#ffd400"
            self.path_item_ids.append(
                self.map_canvas.create_oval(
                    col - radius,
                    row - radius,
                    col + radius,
                    row + radius,
                    outline=color,
                    width=2,
                )
            )

    def _read_demo_points(self):
        start, start_mode = normalize_demo_point([self.start_x.get(), self.start_y.get(), self.start_z.get()])
        end, end_mode = normalize_demo_point([self.end_x.get(), self.end_y.get(), self.end_z.get()])
        if start_mode != "AirSim local" or end_mode != "AirSim local":
            self.log(f"坐标自动转换: start={start_mode}, end={end_mode}")
            self.log(f"转换后 start={start.round(3)}, end={end.round(3)}")
        return start, end

    def launch_airsim(self):
        pipeline = self._get_pipeline()
        if pipeline is None:
            return

        success, msg = pipeline.launch_airsim()
        self.log(msg)
        if success:
            messagebox.showinfo("成功", "AirSim 启动命令已发送。")

    def init_model(self):
        pipeline = self._get_pipeline()
        if pipeline is None:
            return

        self.init_btn.config(state="disabled")

        def task():
            self.log("正在初始化模型/连接 AirSim...")
            success, msg = pipeline.init_navigation()
            self.log(msg)
            self.root.after(0, lambda: self.init_btn.config(state="normal"))
            if success:
                self.root.after(0, lambda: messagebox.showinfo("成功", "模型与导航管线加载完成。"))
            else:
                self.root.after(0, lambda: messagebox.showerror("初始化失败", msg))

        threading.Thread(target=task, daemon=True).start()

    def plan_path(self):
        pipeline = self._get_pipeline()
        if pipeline is None:
            return

        try:
            start, end = self._read_demo_points()
        except ValueError:
            messagebox.showerror("错误", "请输入有效的坐标数值。")
            return

        self.plan_btn.config(state="disabled")
        self.nav_btn.config(state="disabled")
        self.clear_path_overlay()
        self.log(f"开始 A* 规划: {start.round(3)} -> {end.round(3)}")

        def task():
            waypoints, msg = pipeline.plan_path(start, end)
            self.log(msg)
            if waypoints:
                self.waypoints = waypoints
                self.log(f"生成 {len(waypoints)} 个局部目标点，已绘制到右侧地图。")
                self.root.after(0, self.draw_waypoints, waypoints)
                self.root.after(0, lambda: self.nav_btn.config(state="normal"))
            self.root.after(0, lambda: self.plan_btn.config(state="normal"))

        threading.Thread(target=task, daemon=True).start()

    def run_navigation(self):
        pipeline = self._get_pipeline()
        if pipeline is None:
            return

        if not self.waypoints:
            messagebox.showwarning("提示", "请先进行 A* 路径规划。")
            return

        self.nav_btn.config(state="disabled")

        def task():
            self.log("开始 TD3 分段导航...")
            success, msg = pipeline.run_navigation(self.waypoints, status_callback=self.log)
            self.log(msg)
            if success:
                self.root.after(0, lambda: messagebox.showinfo("完成", "任务成功完成。"))
            else:
                self.root.after(0, lambda: messagebox.showerror("失败", msg))
            self.root.after(0, lambda: self.nav_btn.config(state="normal"))

        threading.Thread(target=task, daemon=True).start()


if __name__ == "__main__":
    root = tk.Tk()
    app = NavGUI(root)
    root.mainloop()
