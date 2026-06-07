## Summary
- 在现有 Tkinter 界面 [gui_app.py](file:///e:/Desktop/course%20project/AirSim_RL/Src/gui_app.py) 右侧增加一张 Model6 的 top-down 占据栅格图（本次使用 [topdown_depth_20260601_165808_occupancy_h10_infl2.png](file:///e:/Desktop/course%20project/AirSim_RL/Network/Model6/topdown_depth/topdown_depth_20260601_165808_occupancy_h10_infl2.png)），支持点击图片像素点来自动填充起点/终点的 X/Y 坐标。
- 像素 → 坐标转换逻辑采用 Model6 的 top-down 地图坐标系（row 轴对应 -x，col 轴对应 +y），并读取其元数据 [topdown_depth_20260601_165808_occupancy_h10_infl2.json](file:///e:/Desktop/course%20project/AirSim_RL/Network/Model6/topdown_depth/topdown_depth_20260601_165808_occupancy_h10_infl2.json) 中的 origin_pixel 与 meters_per_pixel。
- 交互方式：单选切换“设置起点/设置终点”；点击地图仅更新 X/Y，Z 保留输入框当前值；地图嵌入主界面右侧。

## Current State Analysis
- 当前 UI 入口为 [gui_app.py](file:///e:/Desktop/course%20project/AirSim_RL/Src/gui_app.py)：仅提供手动输入 start/end 的 (x,y,z) 文本框与按钮触发规划/执行。
- Model6 已提供 top-down 地图的坐标系定义与换算公式（见 [occupancy_planner.py:TopDownOccupancyGrid](file:///e:/Desktop/course%20project/AirSim_RL/Network/Model6/occupancy_planner.py#L151-L224)）：
  - origin_pixel(row,col) 对应 AirSim local NED 的 (x=0,y=0)
  - row 轴向下 = -x，col 轴向右 = +y
  - 像素(cell) → 世界坐标：x = (origin_row - row) * meters_per_pixel_x；y = (col - origin_col) * meters_per_pixel_y
- 本次使用的地图文件存在于仓库内，无需依赖 occupancy_tools 绘图脚本：
  - 图片：[topdown_depth_20260601_165808_occupancy_h10_infl2.png](file:///e:/Desktop/course%20project/AirSim_RL/Network/Model6/topdown_depth/topdown_depth_20260601_165808_occupancy_h10_infl2.png)（390x350）
  - 元数据：[topdown_depth_20260601_165808_occupancy_h10_infl2.json](file:///e:/Desktop/course%20project/AirSim_RL/Network/Model6/topdown_depth/topdown_depth_20260601_165808_occupancy_h10_infl2.json)（origin_pixel: row=170,col=200；meters_per_pixel: x≈1,y=1）

## Proposed Changes
### 1) 在 GUI 增加右侧“地图选点”面板
- 文件：[gui_app.py](file:///e:/Desktop/course%20project/AirSim_RL/Src/gui_app.py)
- 变更点
  - 将整体布局从单列改为左右分栏：
    - 左侧：保留原有“环境控制 / 导航设置 / 操作按钮 / 状态日志”
    - 右侧：新增“地图选点”LabelFrame，内含 Canvas + 地图图片
  - 窗口尺寸调整：由 600x500 改为更适合左右分栏的尺寸（例如 1050x560，具体以图片 390x350 + 控件区宽度为准）
  - 增加单选按钮：
    - “设置起点”
    - “设置终点”
  - Canvas 交互：
    - 绑定 `<Button-1>`：获取点击像素 (col=x, row=y)
    - 使用元数据进行像素→坐标换算，仅更新 X/Y，对应的 Z 不改
    - 在状态栏/日志区输出一行：像素坐标 + 转换后的世界坐标
    - 在地图上绘制起点/终点标记（例如不同颜色的圆点/十字），用于可视化反馈

### 2) 读取 Model6 top-down 元数据并实现像素→坐标换算
- 文件：[gui_app.py](file:///e:/Desktop/course%20project/AirSim_RL/Src/gui_app.py)
- 变更点
  - 增加 `json` 导入
  - 增加一个轻量的元数据加载函数（不依赖 occupancy_tools）：
    - 默认元数据路径：`REPO_ROOT/Network/Model6/topdown_depth/topdown_depth_20260601_165808_occupancy_h10_infl2.json`
    - 从 json 读取：
      - `origin_row = coordinate_frame.origin_pixel.row`
      - `origin_col = coordinate_frame.origin_pixel.col`
      - `mpp_x = meters_per_pixel.x`
      - `mpp_y = meters_per_pixel.y`
  - 换算函数（与 Model6 的 [TopDownOccupancyGrid.cell_to_world](file:///e:/Desktop/course%20project/AirSim_RL/Network/Model6/occupancy_planner.py#L216-L224) 保持一致）：
    - `world_x = (origin_row - row) * mpp_x`
    - `world_y = (col - origin_col) * mpp_y`
  - 边界检查：
    - 若点击超出图片范围则忽略
    - 若元数据/图片缺失则在 UI 中弹窗报错并禁用地图区域

### 3) 图片加载方案（不引入新依赖）
- 文件：[gui_app.py](file:///e:/Desktop/course%20project/AirSim_RL/Src/gui_app.py)
- 变更点
  - 使用 `tk.PhotoImage(file=png_path)` 加载 PNG（不引入 Pillow）
  - 若加载失败，提示用户当前 Python/Tk 版本不支持 PNG，并给出替代方案（安装 Pillow 或将图片转为 GIF/PPM）

## Assumptions & Decisions
- 采用用户已确认的交互偏好：
  - 选点方式：单选切换（设置起点/终点）
  - Z 值：保留输入框当前值，仅更新 X/Y
  - UI 布局：主界面右侧嵌入地图
- 坐标系以 Model6 top-down 元数据为准：像素行 row 向下对应 -x，像素列 col 向右对应 +y。
- 本次不依赖 `occupancy_tools` 内的绘图脚本，仅复用其产出的 png/json。

## Verification
- 静态验证
  - 确认 [gui_app.py](file:///e:/Desktop/course%20project/AirSim_RL/Src/gui_app.py) 能在无 Pillow 环境下加载 png（PhotoImage）。
  - 使用元数据做 2 个快速自检（运行时打印/断言均可）：
    - 点击 origin_pixel(row=170,col=200) 时，转换结果应接近 (x=0,y=0)
    - 点击 (row=170,col=201) 时，y 应接近 +1（mpp_y=1）
- 手工验证（需要用户本地环境可运行 GUI + AirSim）
  - 运行 `python Src/gui_app.py`，确认右侧显示地图
  - 选择“设置起点”，点击地图任意位置，起点 X/Y 自动更新，Z 不变
  - 选择“设置终点”，点击地图任意位置，终点 X/Y 自动更新，Z 不变
  - 点击规划/执行流程不受影响（仅改变输入方式）
