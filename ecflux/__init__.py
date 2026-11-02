"""ecflux: 面向端云大小模型协同的 FLUX.2 Klein 端侧模型压缩 / 蒸馏 / 路由工具包。

模块划分对应开题报告第四章：
- ecflux.pruning : 4.2 多粒度结构化剪枝 + APRA
- ecflux.distill : 4.3 面向 Rectified Flow 的语义对齐知识蒸馏
- ecflux.router  : 4.4 任务复杂度判别器与端云协同推理
- ecflux.metrics : 表2 评估指标
"""

__version__ = "0.1.0"
