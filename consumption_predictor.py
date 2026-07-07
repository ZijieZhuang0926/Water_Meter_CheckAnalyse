import pandas as pd
import numpy as np
from typing import List, Tuple, Dict, Any, Optional
from collections import Counter


class ConsumptionPredictor:
    """
    水表日用量预测器（整数数据优化版）
    
    核心改进：
    1. 周期性检测：用众数一致性比率替代F-ratio，避免整数数据组内方差为0的问题
    2. MAD下限：设为0.5（半个整数单位），防止整数数据MAD=0导致统计失效
    3. 趋势预测：用EMA替代线性外推，累计值差分不应线性增长
    4. 整数量化补偿：预测区间上下界各扩展±0.5，确保与整数增量比较在同一空间
    
    注意：本模块假设输入的history_readings已经经过清洗，不含异常数据
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.mad_factor = self.config.get('mad_factor', 3.0)
        self.min_samples = self.config.get('min_samples', 8)
        self.min_history_days = self.config.get('min_history_days', 60)
        self.base_weekday_weight = self.config.get('weekday_weight', 0.6)
        self.trend_weight = self.config.get('trend_weight', 0.3)
        self.holiday_weight = self.config.get('holiday_weight', 0.1)
        self.mad_floor = self.config.get('mad_floor', 0.5)
        self.ema_alpha = self.config.get('ema_alpha', 0.3)

    def predict(self, today_date: str, history_readings: List[Tuple[str, float]],
                baseline_reading: Optional[float] = None) -> Dict[str, Any]:
        if len(history_readings) < self.min_history_days:
            return self._build_result(None, None, None, 0.0,
                                      '历史数据不足（需≥{}天）'.format(self.min_history_days),
                                      {'reason': 'insufficient_history', 'history_days': len(history_readings)})

        history_df = pd.DataFrame(history_readings, columns=['date', 'reading'])
        history_df['date'] = pd.to_datetime(history_df['date'])
        history_df = history_df.sort_values('date').reset_index(drop=True)

        if baseline_reading is not None:
            history_df['prev_reading'] = history_df['reading'].shift(1)
            history_df['date_diff'] = (history_df['date'] - history_df['date'].shift(1)).dt.days
            
            first_valid_idx = history_df[history_df['date_diff'].notna()].index[0]
            history_df.loc[first_valid_idx, 'prev_reading'] = baseline_reading
            
            history_df['increment'] = (history_df['reading'] - history_df['prev_reading']) / history_df['date_diff']
        else:
            history_df['prev_reading'] = history_df['reading'].shift(1)
            history_df['date_diff'] = (history_df['date'] - history_df['date'].shift(1)).dt.days
            history_df['increment'] = (history_df['reading'] - history_df['prev_reading']) / history_df['date_diff']
            
        history_df = history_df.dropna(subset=['increment'])
        history_df = history_df[history_df['increment'] > 0]

        if len(history_df) < 7:
            return self._build_result(None, None, None, 0.0,
                                      '有效增量数据不足',
                                      {'reason': 'insufficient_valid_data', 'valid_samples': len(history_df)})

        today = pd.to_datetime(today_date)
        today_weekday = today.weekday()
        
        # is_today_notwork = self._is_holiday_safe(today_date) 
        is_today_holiday = self._is_holiday_safe(today_date)
        is_today_workday = self._is_workday_safe(today_date)

        periodicity_score = self._detect_periodicity_mode(history_df)

        weekday_weight = self.base_weekday_weight * periodicity_score

        weekday_prediction = self._predict_by_weekday(history_df, today_weekday)
        trend_prediction = self._predict_by_ema(history_df)
        holiday_adjustment = self._calculate_holiday_adjustment_safe(history_df, is_today_holiday, is_today_workday)

        predictions = []
        weights = []

        if weekday_prediction['value'] is not None:
            predictions.append(weekday_prediction['value'])
            weights.append(weekday_weight * weekday_prediction['confidence'])

        if trend_prediction['value'] is not None:
            predictions.append(trend_prediction['value'])
            weights.append(self.trend_weight * trend_prediction['confidence'])

        if holiday_adjustment != 1.0:
            base_value = predictions[0] if predictions else history_df['increment'].median()
            predictions.append(base_value * holiday_adjustment)
            weights.append(self.holiday_weight)

        if not predictions:
            return self._build_result(None, None, None, 0.0,
                                      '无法生成预测',
                                      {'reason': 'no_prediction_available'})

        total_weight = sum(weights)
        if total_weight == 0:
            weighted_prediction = np.mean(predictions)
        else:
            weighted_prediction = sum(p * w for p, w in zip(predictions, weights)) / total_weight

        overall_confidence = min(0.95, 0.5 + 0.05 * len(history_df))

        all_increments = history_df['increment'].values
        raw_mad = np.median(np.abs(all_increments - np.median(all_increments)))
        overall_mad = max(raw_mad, self.mad_floor)
        # 计算置信区间
        lower_bound = weighted_prediction - self.mad_factor * overall_mad
        upper_bound = weighted_prediction + self.mad_factor * overall_mad

        lower_bound -= 0.5 # 量化补偿
        upper_bound += 0.5

        lower_bound = max(0, lower_bound)

        metadata = {
            'weekday_prediction': weekday_prediction,
            'trend_prediction': trend_prediction,
            'holiday_adjustment': holiday_adjustment,
            'is_holiday': is_today_holiday,
            'is_workday': is_today_workday,
            'overall_mad': round(overall_mad, 2),
            'raw_mad': round(raw_mad, 2),
            'history_days': len(history_readings),
            'valid_samples': len(history_df),
            'periodicity_score': round(periodicity_score, 2),
            'adjusted_weekday_weight': round(weekday_weight, 2),
            'quantization_compensation': 0.5
        }

        return self._build_result(round(weighted_prediction, 2), round(lower_bound, 2),
                                  round(upper_bound, 2), round(overall_confidence, 2),
                                  '预测完成', metadata)

    def _detect_periodicity_mode(self, history_df: pd.DataFrame) -> float:
        """
        用众数一致性比率检测周期性强度
        
        返回值：0~1，越高表示周期性越强
        原理：统计同星期组内众数占比，占比>60%则周期性强
        """
        weekday_groups = history_df.groupby(history_df['date'].dt.weekday)['increment']

        mode_consistencies = []
        for _, group in weekday_groups:
            if len(group) >= 3:
                counts = Counter(group.round(0).astype(int))
                if counts:
                    max_count = max(counts.values())
                    consistency = max_count / len(group)
                    mode_consistencies.append(consistency)

        if not mode_consistencies:
            return 0.5

        avg_consistency = np.mean(mode_consistencies)

        score = min(1.0, avg_consistency * 1.5)

        score = max(0.3, min(1.0, score))

        return score

    def _predict_by_weekday(self, history_df: pd.DataFrame, today_weekday: int) -> Dict[str, Any]:
        same_weekday_df = history_df[history_df['date'].dt.weekday == today_weekday]
        sample_count = len(same_weekday_df)

        if sample_count < self.min_samples:
            return {'value': None, 'confidence': 0.0, 'sample_count': sample_count, 'reason': '样本不足'}

        increments = same_weekday_df['increment'].values
        median = np.median(increments)

        confidence = min(0.9, 0.5 + 0.05 * sample_count)

        return {
            'value': round(median, 2),
            'confidence': round(confidence, 2),
            'sample_count': sample_count,
            'median': round(median, 2),
            'reason': '同星期中位数'
        }

    def _predict_by_ema(self, history_df: pd.DataFrame) -> Dict[str, Any]:
        """
        用指数移动平均(EMA)预测短期趋势
        
        优点：捕捉平移变化，不假设线性增长，适合累计值差分场景
        """
        recent_df = history_df.tail(14)
        if len(recent_df) < 3:
            return {'value': None, 'confidence': 0.0, 'sample_count': len(recent_df), 'reason': '近期数据不足'}

        recent_increments = recent_df['increment'].values

        ema_values = []
        ema = recent_increments[0]
        ema_values.append(ema)
        for val in recent_increments[1:]:
            ema = self.ema_alpha * val + (1 - self.ema_alpha) * ema
            ema_values.append(ema)

        final_ema = ema_values[-1]

        confidence = min(0.8, 0.4 + 0.05 * len(recent_df))

        return {
            'value': round(final_ema, 2),
            'confidence': round(confidence, 2),
            'sample_count': len(recent_df),
            'ema_alpha': self.ema_alpha,
            'reason': 'EMA指数移动平均'
        }

    def _calculate_holiday_adjustment_safe(self, history_df: pd.DataFrame,
                                           is_holiday: bool, is_workday: bool) -> float:
        try:
            from chinese_calendar import is_holiday as check_holiday, is_workday as check_workday

            if not is_holiday and is_workday:
                return 1.0

            if is_holiday:
                holiday_df = history_df[history_df['date'].apply(
                    lambda d: check_holiday(d.date()) if hasattr(d, 'date') else False
                )]
                workday_df = history_df[history_df['date'].apply(
                    lambda d: check_workday(d.date()) if hasattr(d, 'date') else False
                )]

                if len(holiday_df) >= 3 and len(workday_df) >= 3:
                    holiday_median = np.median(holiday_df['increment'].values)
                    workday_median = np.median(workday_df['increment'].values)
                    if workday_median > 0:
                        return round(holiday_median / workday_median, 2)

                return 0.8

            if not is_workday and not is_holiday:
                weekend_df = history_df[history_df['date'].dt.weekday >= 5]
                weekday_df = history_df[history_df['date'].dt.weekday < 5]

                if len(weekend_df) >= 3 and len(weekday_df) >= 3:
                    weekend_median = np.median(weekend_df['increment'].values)
                    weekday_median = np.median(weekday_df['increment'].values)
                    if weekday_median > 0:
                        return round(weekend_median / weekday_median, 2)

                return 0.9

        except Exception:
            pass

        return 1.0

    def _is_holiday_safe(self, date_str: str) -> bool:
        try:
            from chinese_calendar import is_holiday
            date = pd.to_datetime(date_str).date()
            return is_holiday(date)
        except Exception:
            return False

    def _is_workday_safe(self, date_str: str) -> bool:
        try:
            from chinese_calendar import is_workday
            date = pd.to_datetime(date_str).date()
            return is_workday(date)
        except Exception:
            return True

    def _build_result(self, predicted_value: Optional[float], lower_bound: Optional[float],
                      upper_bound: Optional[float], confidence: float,
                      details: str, metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        return {
            'predicted_value': predicted_value,
            'lower_bound': lower_bound,
            'upper_bound': upper_bound,
            'confidence': confidence,
            'details': details,
            'metadata': metadata or {}
        }
