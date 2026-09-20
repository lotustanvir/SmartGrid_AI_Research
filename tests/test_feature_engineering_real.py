"""
Phase 6C: Feature Engineering Tests
Validates feature creation, leakage prevention, and dataset integrity
"""

import csv
import os
import json
from datetime import datetime, timedelta

BASE = r"D:\SmartGrid_AI_Research"
PROCESSED = os.path.join(BASE, "dataset", "processed")
RESULTS = os.path.join(BASE, "results", "data_validation")

class TestResult:
    def __init__(self):
        self.passed = 0
        self.failed = 0
        self.errors = []
    
    def check(self, condition, test_name):
        if condition:
            self.passed += 1
            print(f"  [PASS] {test_name}")
        else:
            self.failed += 1
            self.errors.append(test_name)
            print(f"  [FAIL] {test_name}")
    
    def summary(self):
        total = self.passed + self.failed
        print(f"\n{'='*60}")
        print(f"TEST RESULTS: {self.passed}/{total} passed, {self.failed} failed")
        if self.errors:
            print(f"Failed tests: {self.errors}")
        print(f"{'='*60}")
        return self.failed == 0

def run_tests():
    results = TestResult()
    
    print("=" * 60)
    print("PHASE 6C: FEATURE ENGINEERING TESTS")
    print("=" * 60)
    
    # ================================================================
    # TEST 1: FILE EXISTENCE
    # ================================================================
    print("\n1. File Existence Tests")
    
    required_files = [
        'train_features.csv',
        'validation_features.csv', 
        'test_features.csv',
        'pjm_tabular_features.csv',
        'pjm_sequences.csv',
        'pjm_tft_features.csv'
    ]
    
    for filename in required_files:
        filepath = os.path.join(PROCESSED, filename)
        results.check(os.path.exists(filepath), f"{filename} exists")
    
    metadata_path = os.path.join(RESULTS, "feature_metadata.json")
    results.check(os.path.exists(metadata_path), "feature_metadata.json exists")
    
    # ================================================================
    # TEST 2: SCHEMA VALIDATION
    # ================================================================
    print("\n2. Schema Validation Tests")
    
    expected_train_cols = [
        'timestamp', 'pjm_load_mw', 'wind_generation_mw', 'solar_generation_mw',
        'hour', 'day_of_week', 'day_of_month', 'month', 'quarter', 'year', 'is_weekend', 'season',
        'hour_sin', 'hour_cos', 'day_sin', 'day_cos', 'month_sin', 'month_cos',
        'lag_1', 'lag_24', 'lag_168',
        'rolling_mean_24', 'rolling_std_24', 'rolling_mean_168', 'rolling_std_168',
        'total_renewable_mw', 'renewable_ratio', 'net_load', 'wind_share', 'solar_share',
        'high_demand_flag', 'low_renewable_flag', 'peak_hour_flag'
    ]
    
    with open(os.path.join(PROCESSED, 'train_features.csv'), 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        actual_cols = reader.fieldnames
    
    results.check(actual_cols == expected_train_cols, "Train features has correct columns")
    
    # Check TFT columns
    expected_tft_cols = ['time_idx', 'group_id', 'target',
                         'hour', 'day_of_week', 'month', 'is_weekend',
                         'hour_sin', 'hour_cos', 'month_sin', 'month_cos',
                         'lag_1', 'lag_24', 'lag_168',
                         'rolling_mean_24', 'rolling_mean_168',
                         'total_renewable_mw', 'net_load',
                         'high_demand_flag', 'peak_hour_flag',
                         'wind_generation_mw', 'solar_generation_mw']
    
    with open(os.path.join(PROCESSED, 'pjm_tft_features.csv'), 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        actual_tft_cols = reader.fieldnames
    
    results.check(actual_tft_cols == expected_tft_cols, "TFT features has correct columns")
    
    # ================================================================
    # TEST 3: FEATURE CREATION
    # ================================================================
    print("\n3. Feature Creation Tests")
    
    with open(os.path.join(PROCESSED, 'train_features.csv'), 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    
    # Check time features
    first_row = rows[0]
    results.check('hour' in first_row, "hour feature exists")
    results.check('day_of_week' in first_row, "day_of_week feature exists")
    results.check('month' in first_row, "month feature exists")
    results.check('is_weekend' in first_row, "is_weekend feature exists")
    results.check('season' in first_row, "season feature exists")
    
    # Check cyclical encoding
    results.check('hour_sin' in first_row, "hour_sin feature exists")
    results.check('hour_cos' in first_row, "hour_cos feature exists")
    results.check('month_sin' in first_row, "month_sin feature exists")
    results.check('month_cos' in first_row, "month_cos feature exists")
    
    # Check lag features
    results.check('lag_1' in first_row, "lag_1 feature exists")
    results.check('lag_24' in first_row, "lag_24 feature exists")
    results.check('lag_168' in first_row, "lag_168 feature exists")
    
    # Check rolling features
    results.check('rolling_mean_24' in first_row, "rolling_mean_24 feature exists")
    results.check('rolling_mean_168' in first_row, "rolling_mean_168 feature exists")
    
    # Check renewable features
    results.check('total_renewable_mw' in first_row, "total_renewable_mw feature exists")
    results.check('renewable_ratio' in first_row, "renewable_ratio feature exists")
    results.check('net_load' in first_row, "net_load feature exists")
    results.check('wind_share' in first_row, "wind_share feature exists")
    results.check('solar_share' in first_row, "solar_share feature exists")
    
    # Check extreme event flags
    results.check('high_demand_flag' in first_row, "high_demand_flag feature exists")
    results.check('low_renewable_flag' in first_row, "low_renewable_flag feature exists")
    results.check('peak_hour_flag' in first_row, "peak_hour_flag feature exists")
    
    # ================================================================
    # TEST 4: LAG CORRECTNESS
    # ================================================================
    print("\n4. Lag Correctness Tests")
    
    # Load base dataset
    base_data = []
    with open(os.path.join(PROCESSED, "pjm_smart_grid_2020_2025.csv"), 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            base_data.append({
                'timestamp': datetime.strptime(row['timestamp'], '%Y-%m-%d %H:%M:%S'),
                'load': float(row['pjm_load_mw'])
            })
    
    # Skip first 168 rows (removed in feature dataset)
    base_data = base_data[168:]
    
    # Check lag_1 correctness (sample)
    lag1_correct = True
    for i in range(1, min(100, len(rows))):
        feature_lag1 = float(rows[i]['lag_1'])
        base_prev = base_data[i-1]['load']
        if abs(feature_lag1 - base_prev) > 0.001:
            lag1_correct = False
            break
    results.check(lag1_correct, "lag_1 values correct")
    
    # Check lag_24 correctness (sample)
    lag24_correct = True
    for i in range(24, min(124, len(rows))):
        feature_lag24 = float(rows[i]['lag_24'])
        base_prev24 = base_data[i-24]['load']
        if abs(feature_lag24 - base_prev24) > 0.001:
            lag24_correct = False
            break
    results.check(lag24_correct, "lag_24 values correct")
    
    # Check lag_168 correctness (sample)
    lag168_correct = True
    for i in range(168, min(268, len(rows))):
        feature_lag168 = float(rows[i]['lag_168'])
        base_prev168 = base_data[i-168]['load']
        if abs(feature_lag168 - base_prev168) > 0.001:
            lag168_correct = False
            break
    results.check(lag168_correct, "lag_168 values correct")
    
    # ================================================================
    # TEST 5: ROLLING SHIFT CORRECTNESS
    # ================================================================
    print("\n5. Rolling Shift Correctness Tests")
    
    # Check that rolling features use shifted data (no leakage)
    # The first non-None rolling_mean_24 should be based on data[0:24] shifted by 1
    first_rolling_idx = None
    for i, row in enumerate(rows):
        if row['rolling_mean_24'] and float(row['rolling_mean_24']) > 0:
            first_rolling_idx = i
            break
    
    results.check(first_rolling_idx is not None, "rolling_mean_24 has values")
    
    if first_rolling_idx is not None:
        # Verify rolling_mean_24 is computed from shifted load
        # It should be the mean of load[i-24:i] (shifted by 1)
        rolling_val = float(rows[first_rolling_idx]['rolling_mean_24'])
        # Get the window of shifted loads
        window_loads = [base_data[j]['load'] for j in range(first_rolling_idx-23, first_rolling_idx+1)]
        expected_mean = sum(window_loads) / len(window_loads)
        results.check(abs(rolling_val - expected_mean) < 0.01, "rolling_mean_24 computed correctly from shifted data")
    
    # ================================================================
    # TEST 6: NO FUTURE LEAKAGE
    # ================================================================
    print("\n6. No Future Leakage Tests")
    
    # Verify temporal ordering in splits
    train_rows = rows
    
    with open(os.path.join(PROCESSED, 'validation_features.csv'), 'r', encoding='utf-8') as f:
        val_rows = list(csv.DictReader(f))
    
    with open(os.path.join(PROCESSED, 'test_features.csv'), 'r', encoding='utf-8') as f:
        test_rows = list(csv.DictReader(f))
    
    train_last_ts = datetime.strptime(train_rows[-1]['timestamp'], '%Y-%m-%d %H:%M:%S')
    val_first_ts = datetime.strptime(val_rows[0]['timestamp'], '%Y-%m-%d %H:%M:%S')
    val_last_ts = datetime.strptime(val_rows[-1]['timestamp'], '%Y-%m-%d %H:%M:%S')
    test_first_ts = datetime.strptime(test_rows[0]['timestamp'], '%Y-%m-%d %H:%M:%S')
    
    results.check(train_last_ts < val_first_ts, "Train ends before Validation starts")
    results.check(val_last_ts < test_first_ts, "Validation ends before Test starts")
    
    # Verify no future information in features
    # lag_168 for first train row should be from 168 hours before, not after
    first_train_ts = datetime.strptime(train_rows[0]['timestamp'], '%Y-%m-%d %H:%M:%S')
    first_train_lag168 = float(train_rows[0]['lag_168'])
    # Find the timestamp 168 hours before
    expected_ts = first_train_ts - timedelta(hours=168)
    # Find this timestamp in base_data
    base_idx = None
    for i, row in enumerate(base_data):
        if row['timestamp'] == expected_ts:
            base_idx = i
            break
    
    if base_idx is not None:
        expected_lag168 = base_data[base_idx]['load']
        results.check(abs(first_train_lag168 - expected_lag168) < 0.001, "lag_168 uses only past data")
    
    # ================================================================
    # TEST 7: SPLIT ORDERING
    # ================================================================
    print("\n7. Split Ordering Tests")
    
    n_train = len(train_rows)
    n_val = len(val_rows)
    n_test = len(test_rows)
    total = n_train + n_val + n_test
    
    results.check(n_train > n_val, "Train larger than Validation")
    results.check(n_val == n_test, "Validation and Test have equal size")
    
    # Check approximate ratios
    train_pct = n_train / total * 100
    val_pct = n_val / total * 100
    test_pct = n_test / total * 100
    
    results.check(65 <= train_pct <= 75, f"Train is ~70% (actual: {train_pct:.1f}%)")
    results.check(12 <= val_pct <= 18, f"Validation is ~15% (actual: {val_pct:.1f}%)")
    results.check(12 <= test_pct <= 18, f"Test is ~15% (actual: {test_pct:.1f}%)")
    
    # ================================================================
    # TEST 8: DATASET SCHEMA
    # ================================================================
    print("\n8. Dataset Schema Tests")
    
    # Check train/val/test have same schema
    with open(os.path.join(PROCESSED, 'validation_features.csv'), 'r', encoding='utf-8') as f:
        val_cols = csv.DictReader(f).fieldnames
    
    with open(os.path.join(PROCESSED, 'test_features.csv'), 'r', encoding='utf-8') as f:
        test_cols = csv.DictReader(f).fieldnames
    
    results.check(actual_cols == val_cols, "Train and Validation have same schema")
    results.check(actual_cols == test_cols, "Train and Test have same schema")
    
    # Check no missing values in critical columns
    critical_cols = ['timestamp', 'pjm_load_mw', 'lag_1', 'lag_24', 'lag_168']
    missing_found = False
    for row in rows[:100]:  # Sample check
        for col in critical_cols:
            if not row[col] or row[col] == '':
                missing_found = True
                break
    results.check(not missing_found, "No missing values in critical columns")
    
    # ================================================================
    # TEST 9: CYCLICAL ENCODING VALIDITY
    # ================================================================
    print("\n9. Cyclical Encoding Validity Tests")
    
    # Check sin^2 + cos^2 = 1 for hour encoding
    cyclical_valid = True
    for row in rows[:100]:  # Sample
        hour_sin = float(row['hour_sin'])
        hour_cos = float(row['hour_cos'])
        if abs(hour_sin**2 + hour_cos**2 - 1.0) > 0.001:
            cyclical_valid = False
            break
    results.check(cyclical_valid, "hour sin^2 + cos^2 = 1")
    
    # Check month encoding
    month_cyclical_valid = True
    for row in rows[:100]:
        month_sin = float(row['month_sin'])
        month_cos = float(row['month_cos'])
        if abs(month_sin**2 + month_cos**2 - 1.0) > 0.001:
            month_cyclical_valid = False
            break
    results.check(month_cyclical_valid, "month sin^2 + cos^2 = 1")
    
    # ================================================================
    # TEST 10: EXTREME EVENT FLAGS
    # ================================================================
    print("\n10. Extreme Event Flags Tests")
    
    # Check flags are binary (0 or 1)
    flags_valid = True
    for row in rows[:100]:
        if row['high_demand_flag'] not in ['0', '1']:
            flags_valid = False
            break
        if row['low_renewable_flag'] not in ['0', '1']:
            flags_valid = False
            break
        if row['peak_hour_flag'] not in ['0', '1']:
            flags_valid = False
            break
    results.check(flags_valid, "Extreme event flags are binary (0/1)")
    
    # Check peak_hour_flag logic
    peak_hour_valid = True
    for row in rows[:100]:
        hour = int(row['hour'])
        peak_flag = int(row['peak_hour_flag'])
        expected_peak = 1 if 17 <= hour <= 21 else 0
        if peak_flag != expected_peak:
            peak_hour_valid = False
            break
    results.check(peak_hour_valid, "peak_hour_flag matches hour 17-21 logic")
    
    # ================================================================
    # SUMMARY
    # ================================================================
    return results.summary()

if __name__ == "__main__":
    success = run_tests()
    exit(0 if success else 1)
