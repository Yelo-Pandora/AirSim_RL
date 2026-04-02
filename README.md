# AirSim_RL 项目



本项目是一个基于 Microsoft AirSim 仿真环境的深度强化学习（DRL）研究平台。项目集成了环境交互、数据传输处理以及多种深度强化学习模型的实现。



## 项目文件结构说明



```text

AirSim_RL/

├── Data/                 # 数据集目录：预计用来存放采集到的训练数据、评估数据或预处理后的数据集

├── Network/              # 模型目录：存放各种深度强化学习算法的实现

│   ├── Model1/           # 具体的模型实现 1

│   └── Model2/           # 具体的模型实现 2

├── Transmission/         # 数据处理目录：存放数据获取、传输与预处理脚本

├── README.md             # 项目说明文档

├── .gitignore            # ignore文件，暂定用来忽略exe和IDE相关配置文件

└── [AirSim_Env].exe      # AirSim 仿真环境的可执行文件（暂定存放在根目录）

```

