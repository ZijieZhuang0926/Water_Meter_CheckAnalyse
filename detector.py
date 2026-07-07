import pandas as pd
import numpy as np
from typing import List, Tuple, Dict, Any, Optional
from collections import Counter

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from hard_rules import HardRuleChecker
from consumption_predictor import ConsumptionPredictor


class WaterMeterAnomalyDetector:
    """
    水表读数异常检测器（管道版）
    
    架构：三层管道，逐层过滤
    ┌─────────────────────────────────────────────────────────┐
    │  第0层：数据清洗管道 (_clean_history)                    │
    │  - 用hard_rules逐条扫描历史，标记异常读数                 │
    │  - 输出：清洗后的历史数据 + 可信基线                      │
    ├─────────────────────────────────────────────────────────┤
    │  第一层：硬规则检测 (HardRuleChecker)                    │
    │  - 用清洗后的基线检测今日读数                             │
    │  - 违反 → 直接判定异常                                   │
    ├─────────────────────────────────────────────────────────┤
    │  第二层：ML预测检测 (ConsumptionPredictor)               │
    │  - 接收已清洗的历史数据做预测                             │
    │  - 实际增量超出预测区间 → 异常                            │
    └─────────────────────────────────────────────────────────┘
    
    渐进式检测：
    - 0-59天：仅硬规则检测（冷启动阶段）
    - 60天以上：启用ML预测（完整检测阶段）
    """

    ANOMALY_TYPES = {
        'hard_rule_violation': '硬规则违反',
        'prediction_deviation': '预测偏差',
        'normal': '正常'
    }

    DETECTION_STAGES = {
        0: {'name': '冷启动阶段', 'description': '0-59天，仅硬规则检测'},
        1: {'name': '完整检测阶段', 'description': '60天以上，启用ML预测'}
    }

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        
        self.hard_rules = HardRuleChecker()
        
        predictor_config = {
            'mad_factor': self.config.get('mad_factor', 3.0),
            'min_samples': self.config.get('min_samples', 8),
            'min_history_days': self.config.get('min_history_days', 60),
            'weekday_weight': self.config.get('weekday_weight', 0.6),
            'trend_weight': self.config.get('trend_weight', 0.3),
            'holiday_weight': self.config.get('holiday_weight', 0.1)
        }
        self.predictor = ConsumptionPredictor(predictor_config)

    def _clean_history(self, history_readings: List[Tuple[str, float]]) -> Tuple[List[Tuple[str, float]], Optional[Tuple[str, float]]]:
        """
        数据清洗管道：用hard_rules扫描历史，标记异常读数
        
        返回：(清洗后的历史数据, 可信基线)
        
        清洗规则：
        1. 负增量 → 异常
        2. 字轮进位异常 → 异常
        3. 增量超出MAD阈值 → 异常
        4. 读数偏离正常组中位数太远 → 异常
        5. 位数异常 → 异常
        """
        if not history_readings:
            return [], None

        history_df = pd.DataFrame(history_readings, columns=['date', 'reading'])
        history_df['date'] = pd.to_datetime(history_df['date'])
        history_df = history_df.sort_values('date').reset_index(drop=True)

        history_df['prev_reading'] = history_df['reading'].shift(1)
        history_df['date_diff'] = (history_df['date'] - history_df['date'].shift(1)).dt.days
        history_df['increment'] = (history_df['reading'] - history_df['prev_reading']) / history_df['date_diff']
        history_df = history_df.dropna(subset=['increment']).reset_index(drop=True)

        history_df['digits'] = history_df['reading'].apply(
            lambda x: len(str(int(x)).lstrip('0')) if x > 0 else 1
        )

        history_df['is_anomaly'] = False

        for i in range(1, len(history_df)):
            current_row = history_df.iloc[i]
            prev_row = history_df.iloc[i - 1]
            
            total_increment = current_row['reading'] - prev_row['reading']
            wheel_result = self.hard_rules._check_wheel_digit_anomaly(prev_row['reading'], current_row['reading'], total_increment)
            has_wheel_anomaly = wheel_result['is_anomaly']
            has_negative_increment = current_row['increment'] < 0
            
            if has_wheel_anomaly or has_negative_increment:
                history_df.loc[i, 'is_anomaly'] = True

        positive_increments = history_df[(history_df['increment'] > 0) & (~history_df['is_anomaly'])]['increment'].values
        if len(positive_increments) >= 3:
            median_inc = np.median(positive_increments)
            mad = np.median(np.abs(positive_increments - median_inc))
            mad = max(mad, 0.5)
            anomaly_threshold = median_inc + 5 * mad
        else:
            anomaly_threshold = float('inf')
            for i in range(1, len(history_df)):
                if not history_df.loc[i, 'is_anomaly'] and history_df.loc[i, 'increment'] > anomaly_threshold:
                    history_df.loc[i, 'is_anomaly'] = True

        digit_groups = history_df.groupby('digits')
        max_group_digits = None
        max_group_size = 0
        for digits, group in digit_groups:
            if len(group) > max_group_size:
                max_group_size = len(group)
                max_group_digits = digits
        
        if max_group_digits is not None and max_group_size >= 3:
            normal_group = history_df[history_df['digits'] == max_group_digits]
            group_readings = normal_group['reading'].values
            group_median = np.median(group_readings)
            group_mad = np.median(np.abs(group_readings - group_median))
            group_mad = max(group_mad, 1)
            reading_threshold = group_mad * 30
            
            for i in range(len(history_df)):
                if history_df.iloc[i]['is_anomaly']:
                    continue
                if history_df.iloc[i]['digits'] == max_group_digits:
                    if abs(history_df.iloc[i]['reading'] - group_median) > reading_threshold:
                        history_df.loc[history_df.index[i], 'is_anomaly'] = True

        normal_df = history_df[~history_df['is_anomaly']]
        digit_counts = Counter(normal_df['digits'].values)
        total_count = len(normal_df)
        expected_digits = None
        if total_count >= 3:
            most_common_digit, count = digit_counts.most_common(1)[0]
            if count / total_count > 0.5:
                expected_digits = most_common_digit

        prev_normal_reading = None
        for i in range(len(history_df)):
            current_row = history_df.iloc[i]
            
            if not history_df.iloc[i]['is_anomaly']:
                prev_normal_reading = current_row['reading']
            
            if history_df.iloc[i]['is_anomaly']:
                continue
            
            has_digit_anomaly = False
            if expected_digits is not None and current_row['digits'] != expected_digits:
                valid_prev_reading = prev_normal_reading if prev_normal_reading is not None else current_row['reading']
                if not (current_row['digits'] == expected_digits + 1 and str(int(valid_prev_reading)).endswith('9')):
                    if not (current_row['digits'] == expected_digits - 1 and str(int(current_row['reading'])).endswith('0')):
                        has_digit_anomaly = True
            
            if has_digit_anomaly:
                history_df.loc[history_df.index[i], 'is_anomaly'] = True
                if prev_normal_reading is not None:
                    prev_normal_reading = None

        filtered_df = history_df[~history_df['is_anomaly']]
        filtered_readings = [
            (row['date'].strftime('%Y-%m-%d'), float(row['reading']))
            for _, row in filtered_df.iterrows()
        ]

        baseline = None
        if len(filtered_df) > 0:
            latest_row = filtered_df.iloc[-1]
            baseline = (latest_row['date'].strftime('%Y-%m-%d'), float(latest_row['reading']))

        return filtered_readings, baseline

    def check(self, meter_id: str, today_readings: List[float], today_date: str,
              history_readings: List[Tuple[str, float]]) -> Dict[str, Any]:
        today_reading = float(today_readings[-1]) if today_readings else 0.0
        metadata = {
            'detection_stage': 0,
            'stage_name': '冷启动阶段',
            'history_days': len(history_readings),
            'today_reading': today_reading,
            'today_reading_count': len(today_readings),
        }

        filtered_readings, baseline = self._clean_history(history_readings)
        metadata['filtered_history_count'] = len(filtered_readings)
        
        baseline_reading = None
        baseline_date = None
        if baseline:
            baseline_date, baseline_reading = baseline
            metadata['baseline_date'] = baseline_date
            metadata['baseline_reading'] = baseline_reading

        hard_result = self.hard_rules.check(today_readings, today_date, filtered_readings, baseline_reading, baseline_date)

        if not hard_result['is_passed']:
            metadata['violation_type'] = hard_result['violation_type']
            metadata.update(hard_result.get('metadata', {}))
            return self._build_result(meter_id, today_date, today_reading, False,
                                      'hard_rule_violation', hard_result['confidence'],
                                      hard_result['details'], metadata)

        if len(history_readings) == 0:
            metadata['stage_name'] = '基线建立阶段'
            return self._build_result(meter_id, today_date, today_reading, True,
                                      'normal', 0.0, '首次上报，建立基线', metadata)

        history_days = len(filtered_readings)
        detection_stage = 1 if history_days >= self.predictor.min_history_days else 0
        metadata['detection_stage'] = detection_stage
        metadata['stage_name'] = self.DETECTION_STAGES[detection_stage]['name']

        if detection_stage == 1:
            prediction = self.predictor.predict(today_date, filtered_readings, baseline_reading)
            
            if prediction['predicted_value'] is not None:
                metadata['prediction'] = prediction

                prev_reading = baseline_reading if baseline_reading else float(filtered_readings[-1][1])
                date_diff = (pd.to_datetime(today_date) - pd.to_datetime(baseline[0])).days if baseline else 1
                
                today_increment = (today_reading - prev_reading) / date_diff

                upper_bound = prediction['upper_bound']
                lower_bound = prediction['lower_bound']

                if today_increment > upper_bound or today_increment < lower_bound:
                    deviation = abs(today_increment - prediction['predicted_value'])
                    threshold = (upper_bound - lower_bound) / 2
                    confidence = min(deviation / threshold, 1.0) if threshold > 0 else 0.9
                    
                    metadata['today_increment'] = round(today_increment, 2)
                    return self._build_result(meter_id, today_date, today_reading, False,
                                              'prediction_deviation', confidence,
                                              f'预测偏差异常：实际增量{today_increment:.2f}，预测区间[{lower_bound:.2f}, {upper_bound:.2f}]',
                                              metadata)

                metadata['today_increment'] = round(today_increment, 2)

        return self._build_result(meter_id, today_date, today_reading, True,
                                  'normal', 1.0, '今日读数正常', metadata)

    def _build_result(self, meter_id: str, today_date: str, today_reading: float,
                      is_normal: bool, anomaly_type: str, confidence: float,
                      details: str, metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        return {
            'meter_id': meter_id,
            'today_date': today_date,
            'today_reading': today_reading,
            'is_normal': is_normal,
            'anomaly_type': anomaly_type,
            'anomaly_description': self.ANOMALY_TYPES.get(anomaly_type, anomaly_type),
            'confidence': round(confidence, 2),
            'details': details,
            'metadata': metadata or {}
        }