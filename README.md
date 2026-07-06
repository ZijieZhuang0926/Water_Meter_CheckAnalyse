# YUNPAI - 云拍器水表异常检测系统

用于检测云拍器上报的水表读数异常，采用规则优先 + 统计辅助的分层架构。

## 项目结构

```
YUNPAI/
├── detector.py           # 检测器主入口，整合硬规则和预测
├── hard_rules.py         # 硬规则检测模块
├── consumption_predictor.py  # ML预测模块
├── test_real_data.py     # 真实数据测试脚本
├── 架构报告.md            # 系统架构详细说明
├── .gitignore            # Git忽略配置
└── data/                 # 测试数据（已忽略）
```

## 核心功能

### 分层检测架构
1. **可信基线查找**：多阶段异常检测，跳过异常读数找到可信基线
2. **硬规则检测**：读数回退、字轮进位、位数突变、物理上限等
3. **ML预测检测**：同星期基线 + EMA趋势 + 节假日调整

### 渐进式检测策略
- 0天：基线建立阶段
- 1-59天：冷启动阶段（仅硬规则）
- 60天以上：完整检测阶段（硬规则 + ML预测）

## 快速开始

```python
from detector import WaterMeterAnomalyDetector

detector = WaterMeterAnomalyDetector()

result = detector.check(
    meter_id='meter_001',
    today_readings=[24720.0],
    today_date='2026-06-27',
    history_readings=[('2026-06-24', 24648.0)]
)
```

## 运行测试

```bash
python test_real_data.py
```

## 技术特性

- 整数数据优化（MAD下限、众数一致性比率）
- 字轮进位逻辑检测（基于总步数推断期望值）
- 可信基线机制（异常跳变后自动回退）
- 渐进式检测（根据历史数据量动态启用检测层）