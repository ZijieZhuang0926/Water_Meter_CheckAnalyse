import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import numpy as np
from detector import WaterMeterAnomalyDetector


def load_data(file_path):
    df = pd.read_excel(file_path)
    df['设备拍照时间'] = pd.to_datetime(df['设备拍照时间'])
    df = df.sort_values('设备拍照时间').reset_index(drop=True)
    return df


def run_test(detector, meter_id, df):
    results = []
    
    for i in range(1, len(df)):
        today_row = df.iloc[i]
        today_date = today_row['设备拍照时间'].strftime('%Y-%m-%d')
        today_reading = float(today_row['上报数据（m³）'])
        
        history_df = df.iloc[:i]
        history_readings = [
            (row['设备拍照时间'].strftime('%Y-%m-%d'), float(row['上报数据（m³）']))
            for _, row in history_df.iterrows()
        ]
        
        result = detector.check(meter_id, [today_reading], today_date, history_readings)
        results.append(result)
    
    return results


def analyze_results(df, results):
    print("=" * 80)
    print("检测结果分析")
    print("=" * 80)
    
    anomaly_count = sum(1 for r in results if not r['is_normal'])
    normal_count = len(results) - anomaly_count
    
    print(f"\n总检测天数: {len(results)}")
    print(f"正常: {normal_count} 天")
    print(f"异常: {anomaly_count} 天")
    
    print("\n" + "=" * 80)
    print("每日检测详情")
    print("=" * 80)
    
    headers = ["日期", "读数", "增量", "阶段", "是否正常", "异常类型", "置信度", "详情"]
    print(f"{headers[0]:<12} {headers[1]:<10} {headers[2]:<6} {headers[3]:<12} {headers[4]:<8} {headers[5]:<20} {headers[6]:<6} {headers[7]}")
    print("-" * 120)
    
    for i, (result, prev_row, curr_row) in enumerate(zip(results, df.iloc[:-1].values, df.iloc[1:].values)):
        increment = curr_row[3] - prev_row[3]
        date = curr_row[2].strftime('%Y-%m-%d')
        reading = curr_row[3]
        is_normal = "✅" if result['is_normal'] else "❌"
        stage = result['metadata']['stage_name']
        
        print(f"{date:<12} {reading:<10} {increment:<6} {stage:<12} {is_normal:<8} {result['anomaly_type']:<20} {result['confidence']:<6.2f} {result['details']}")
    
    return results


def main():
    print("=" * 80)
    print("云拍器水表异常检测 - 真实数据验证")
    print("=" * 80)
    
    detector = WaterMeterAnomalyDetector({
        'mad_factor': 3.0,
        'min_samples': 8,
        'min_history_days': 60,
        'mad_floor': 0.5,
        'outlier_factor': 5.0
    })
    
    print("\n--- 测试1：无异常数据（八里湖院区-八里湖5号楼-2层）---")
    df_normal = load_data('D:\Vscode\.venv\Energy_Prediction\YUNPAI\data\云拍详情页数据_无异常.xlsx')
    print(f"数据日期范围: {df_normal['设备拍照时间'].min().strftime('%Y-%m-%d')} ~ {df_normal['设备拍照时间'].max().strftime('%Y-%m-%d')}")
    print(f"数据条数: {len(df_normal)}")
    results_normal = run_test(detector, '八里湖5号楼-2层', df_normal)
    analyze_results(df_normal, results_normal)
    
    print("\n\n--- 测试2：有异常数据（八里湖院区-八里湖10号楼）---")
    df_anomaly = load_data('D:\Vscode\.venv\Energy_Prediction\YUNPAI\data\云拍详情页数据_有异常.xlsx')
    print(f"数据日期范围: {df_anomaly['设备拍照时间'].min().strftime('%Y-%m-%d')} ~ {df_anomaly['设备拍照时间'].max().strftime('%Y-%m-%d')}")
    print(f"数据条数: {len(df_anomaly)}")
    
    print("\n异常数据预览（按时间排序）：")
    for _, row in df_anomaly.iterrows():
        print(f"  {row['设备拍照时间'].strftime('%Y-%m-%d')}: {row['上报数据（m³）']}")
    
    results_anomaly = run_test(detector, '八里湖10号楼', df_anomaly)
    analyze_results(df_anomaly, results_anomaly)
    
    print("\n" + "=" * 80)
    print("测试总结")
    print("=" * 80)
    
    normal_anomaly_count = sum(1 for r in results_normal if not r['is_normal'])
    anomaly_anomaly_count = sum(1 for r in results_anomaly if not r['is_normal'])
    
    print(f"\n无异常数据集：")
    print(f"  检测天数: {len(results_normal)}")
    print(f"  误报数: {normal_anomaly_count}")
    print(f"  误报率: {(normal_anomaly_count / len(results_normal) * 100):.1f}%")
    
    print(f"\n有异常数据集：")
    print(f"  检测天数: {len(results_anomaly)}")
    print(f"  检测到异常: {anomaly_anomaly_count}")
    print(f"  检测率: {(anomaly_anomaly_count / len(results_anomaly) * 100):.1f}%")
    
    print("\n详细分析有异常数据中的异常检测：")
    for i, (result, row) in enumerate(zip(results_anomaly, df_anomaly.iloc[1:].values)):
        if not result['is_normal']:
            print(f"\n  日期: {row[2].strftime('%Y-%m-%d')}")
            print(f"  读数: {row[3]}")
            print(f"  异常类型: {result['anomaly_type']}")
            print(f"  详情: {result['details']}")
            print(f"  置信度: {result['confidence']}")
            if 'violation_type' in result['metadata']:
                print(f"  违反规则: {result['metadata']['violation_type']}")


if __name__ == '__main__':
    main()
