import pandas as pd
import numpy as np
from typing import List, Tuple, Dict, Any, Optional


class HardRuleChecker:
    """
    水表硬规则异常检测器（最终版）

    设计原则：
    - 位数突变检测：只检测"是否漏读/多读一位"，不限制增量大小
    - 增量大小异常：交给预测器处理
    - 字轮检测：只检测"数字变化是否符合机械进位规律"

    检测层次：
    1. 数据完整性 → 2. 同日一致性 → 3. 首次上报 → 4. 位数突变
    → 5. 读数回退 → 6. 日期间隔 → 7. 物理上限 → 8. 零增量
    → 9. 字轮进位 → 10. 通过（进入预测器）
    """

    VIOLATION_TYPES = {
        'data_insufficient': '数据不足',
        'negative_increment': '负增量',
        'zero_increment': '零增量（停表）',
        'wheel_digit_mismatch': '字轮进位异常',
        'digit_length_mismatch': '位数突变（漏读/多读）',
        'excessive_date_gap': '日期间隔异常',
        'consecutive_zero_increment': '连续零增量',
        'same_day_inconsistency': '同日读数不一致',
        'normal': '正常'
    }

    PHYSICAL_MAX_DAILY_INCREMENT = 1500

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.physical_max_daily = self.config.get('physical_max_daily', self.PHYSICAL_MAX_DAILY_INCREMENT)
        self.zero_threshold = self.config.get('zero_threshold', 0.01)
        self.consecutive_zero_days = self.config.get('consecutive_zero_days', 3)
        self.same_day_cv_threshold = self.config.get('same_day_cv_threshold', 0.1)
        self.max_date_gap_days = self.config.get('max_date_gap_days', 7)
        
    def check(self, today_readings: List[float], today_date: str,
              history_readings: List[Tuple[str, float]],
              baseline_reading: Optional[float] = None) -> Dict[str, Any]:
        if len(today_readings) == 0:
            return self._build_result(False, 'data_insufficient', 1.0, '今日无读数')

        today_reading = float(today_readings[-1])

        if len(today_readings) > 1:
            consistency_result = self._check_same_day_consistency(today_readings)
            if not consistency_result['is_passed']:
                return consistency_result

        if len(history_readings) == 0:
            return self._build_result(True, 'normal', 0.0, '首次上报')

        history_df = self._prepare_history(history_readings)

        if baseline_reading is not None:
            prev_reading = baseline_reading
            latest_history = history_df.iloc[-1]
            prev_date = latest_history['date']
        else:
            latest_history = history_df.iloc[-1]
            prev_date = latest_history['date']
            prev_reading = float(latest_history['reading'])

        date_diff = (pd.to_datetime(today_date) - prev_date).days

        digit_result = self._check_digit_length(prev_reading, today_reading)
        if not digit_result['is_passed']:
            return digit_result

        if today_reading < prev_reading:
            return self._build_result(False, 'negative_increment', 1.0,
                                      f'读数回退：今日{today_reading}，上次{prev_reading}')

        if date_diff > self.max_date_gap_days:
            return self._build_result(False, 'excessive_date_gap', 0.9,
                                      f'日期间隔异常：间隔{date_diff}天',
                                      {'date_gap_days': date_diff})

        if date_diff <= 0:
            return self._build_result(False, 'excessive_date_gap', 1.0,
                                      f'日期异常：间隔{date_diff}天')

        total_increment = today_reading - prev_reading
        today_increment = total_increment / date_diff

        max_allowed = self.physical_max_daily * date_diff
        if total_increment > max_allowed:
            return self._build_result(False, 'wheel_digit_mismatch', 0.95,
                                      f'超出物理上限：总增量{total_increment:.2f}，允许{max_allowed:.2f}',
                                      {'total_increment': total_increment, 'max_allowed': max_allowed})

        if today_increment <= self.zero_threshold:
            consecutive_zero_count = self._check_consecutive_zero_increment(history_df)
            if consecutive_zero_count >= self.consecutive_zero_days - 1:
                return self._build_result(False, 'consecutive_zero_increment', 0.95,
                                          f'连续零增量：已连续{consecutive_zero_count + 1}天',
                                          {'consecutive_zero_days': consecutive_zero_count + 1})
            return self._build_result(False, 'zero_increment', 0.8,
                                      f'零增量：日增量{today_increment:.4f}')

        wheel_result = self._check_wheel_digit_anomaly(prev_reading, today_reading, total_increment)
        if wheel_result['is_anomaly']:
            return self._build_result(False, 'wheel_digit_mismatch',
                                      wheel_result['confidence'], wheel_result['detail'],
                                      {'wheel_digit_suspect_positions': wheel_result['suspect_positions'],
                                       'total_steps': wheel_result.get('total_steps')})

        return self._build_result(True, 'normal', 0.0, '硬规则通过')

    def _prepare_history(self, history_readings: List[Tuple[str, float]]) -> pd.DataFrame:
        df = pd.DataFrame(history_readings, columns=['date', 'reading'])
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date').reset_index(drop=True)
        return df

    def _check_same_day_consistency(self, today_readings: List[float]) -> Dict[str, Any]:
        readings = np.array(today_readings)
        mean_val = np.mean(readings)
        std_val = np.std(readings)

        if mean_val > 0 and std_val / mean_val > self.same_day_cv_threshold:
            cv = round(std_val / mean_val, 3)
            return self._build_result(False, 'same_day_inconsistency', 0.8,
                                      f'同日读数不一致（CV={cv}）',
                                      {'same_day_cv': cv, 'readings': today_readings})

        return self._build_result(True, 'normal', 0.0, '同日读数一致')

    def _check_digit_length(self, prev_reading: float, today_reading: float) -> Dict[str, Any]:
        prev_int = int(prev_reading)
        today_int = int(today_reading)

        prev_str = str(prev_int)
        today_str = str(today_int)

        prev_digits = len(prev_str)
        today_digits = len(today_str)
        digit_diff = today_digits - prev_digits
        actual_increment = today_int - prev_int

        if digit_diff >= 2:
            return self._build_result(False, 'digit_length_mismatch', 0.95,
                                      f'位数突变+{digit_diff}：{prev_reading}→{today_reading}')

        if digit_diff == 1:
            min_increment = (10 ** prev_digits) - prev_int

            if actual_increment < min_increment:
                return self._build_result(False, 'digit_length_mismatch', 0.95,
                                          f'位数突变：增量{actual_increment}不足以使{prev_reading}进位（需≥{min_increment}）')

            return self._build_result(True, 'normal', 0.0, '位数正常（进位）')

        if digit_diff <= -1:
            return self._build_result(False, 'digit_length_mismatch', 0.95,
                                      f'位数减少：{prev_reading}→{today_reading}')

        return self._build_result(True, 'normal', 0.0, '位数正常')

    def _check_consecutive_zero_increment(self, history_df: pd.DataFrame) -> int:
        df = history_df.copy()
        df['prev_reading'] = df['reading'].shift(1)
        df['date_diff'] = (df['date'] - df['date'].shift(1)).dt.days
        df['increment'] = (df['reading'] - df['prev_reading']) / df['date_diff']
        df = df.dropna(subset=['increment'])

        consecutive_count = 0
        max_consecutive = 0
        for _, row in df.iterrows():
            if row['increment'] <= self.zero_threshold:
                consecutive_count += 1
                max_consecutive = max(max_consecutive, consecutive_count)
            else:
                consecutive_count = 0

        return max_consecutive

    def _check_wheel_digit_anomaly(self, yesterday: float, today: float, total_steps: int) -> dict:
        yesterday_int = int(yesterday)
        today_int = int(today)

        if today_int < yesterday_int:
            return {
                'is_anomaly': False,
                'confidence': 0.0,
                'detail': '读数回退，由硬规则处理',
                'suspect_positions': [],
                'total_steps': total_steps
            }

        yesterday_str = str(yesterday_int)
        today_str = str(today_int)
        max_len = max(len(yesterday_str), len(today_str))

        yesterday_str = yesterday_str.zfill(max_len)
        today_str = today_str.zfill(max_len)

        yesterday_digits = [int(c) for c in yesterday_str]
        today_digits = [int(c) for c in today_str]

        expected_today = self._infer_expected_reading(yesterday_digits, total_steps)

        suspect_positions = []
        issues = []

        for i in range(max_len):
            pos_from_right = i + 1
            idx = max_len - 1 - i

            actual = today_digits[idx]
            expected = expected_today[idx]

            if actual != expected:
                suspect_positions.append(pos_from_right)
                issues.append(
                    f"第{pos_from_right}位：昨日{yesterday_digits[idx]}，"
                    f"期望{expected}，实际{actual}（总步数{total_steps}）"
                )

        if not suspect_positions:
            return {
                'is_anomaly': False,
                'confidence': 0.0,
                'detail': '字轮进位逻辑正常',
                'suspect_positions': [],
                'total_steps': total_steps
            }

        max_pos = max(suspect_positions)
        n = len(suspect_positions)

        if max_pos >= 4:
            confidence = min(0.95, 0.80 + 0.05 * n)
        elif max_pos == 3:
            confidence = min(0.90, 0.70 + 0.05 * n)
        elif max_pos == 2:
            confidence = min(0.80, 0.55 + 0.05 * n)
        else:
            confidence = min(0.30, 0.15 + 0.03 * n)

        return {
            'is_anomaly': True,
            'confidence': round(confidence, 2),
            'detail': '；'.join(issues),
            'suspect_positions': suspect_positions,
            'total_steps': total_steps
        }

    def _infer_expected_reading(self, yesterday_digits: List[int], total_steps: int) -> List[int]:
        result = yesterday_digits.copy()
        remaining_steps = total_steps

        for i in range(len(result) - 1, -1, -1):
            if remaining_steps <= 0:
                break

            current = result[i]
            new_value = current + remaining_steps
            result[i] = new_value % 10
            remaining_steps = new_value // 10

        return result

    def _build_result(self, is_passed: bool, violation_type: str,
                      confidence: float, details: str,
                      metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        return {
            'is_passed': is_passed,
            'violation_type': violation_type,
            'violation_description': self.VIOLATION_TYPES.get(violation_type, violation_type),
            'confidence': round(confidence, 2),
            'details': details,
            'metadata': metadata or {}
        }