import time
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
from sklearn.preprocessing import MinMaxScaler

# 匯入自定義的模型模組 (需確保 models.py 與 contrastive.py 在同目錄)
import models
from contrastive import ContrastiveModel

# ==========================================
# 1. 參數與特徵設定設定
# ==========================================
WINDOW_SIZE = 60
STEP_SIZE = 1
DATA_FILE_PATH = 'cab_data_final_train.csv'  # 請替換為實際的檔案路徑

TARGET_FEATURES = [
    'hp_comp_1', 'lp_comp_1', 'comp_current_1', 'cond_current_1', 'fan_current_1',
    'return_air_temp', 'outdoor_temp', 'lp_plate_temp_1', 'superheat_1', 'h_suc_1',
    'heater_temp'
]

SOURCE_TIME_PERIODS = [
    ('2025-04-11 09:14:01', '2025-04-11 09:34:00', 1),
    ('2025-04-11 09:46:01', '2025-04-11 10:06:00', 1),
    ('2025-04-11 10:29:01', '2025-04-11 10:49:00', 2),
    ('2025-04-11 11:01:01', '2025-04-11 11:21:00', 2),
    ('2025-04-11 11:33:01', '2025-04-11 11:53:00', 2),
    ('2025-04-11 12:05:01', '2025-04-11 13:05:00', 0),
    ('2025-04-11 13:45:01', '2025-04-11 14:05:00', 7),
    ('2025-04-11 14:19:01', '2025-04-11 14:39:00', 7),
    ('2025-04-11 14:52:01', '2025-04-11 15:12:00', 7),
    ('2025-04-11 15:46:01', '2025-04-11 16:06:00', 0),
    ('2025-04-14 09:00:01', '2025-04-14 09:40:00', 0),
    ('2025-04-14 10:49:01', '2025-04-14 11:30:00', 8),
    ('2025-04-14 13:25:01', '2025-04-14 14:05:00', 3),
    ('2025-04-14 14:50:01', '2025-04-14 15:30:00', 3),
    ('2025-12-23 10:23:01', '2025-12-23 11:03:00', 0),
    ('2025-12-23 13:19:01', '2025-12-23 13:40:00', 1),
    ('2025-12-23 14:03:01', '2025-12-23 14:23:00', 1),
    ('2025-12-23 15:00:01', '2025-12-23 15:40:00', 3),
    ('2026-01-01 00:00:01', '2026-01-01 01:00:00', 4),
    ('2026-01-01 01:30:01', '2026-01-01 02:30:00', 4),
    ('2026-01-01 03:00:01', '2026-01-01 04:00:00', 4),
    ('2026-01-01 05:00:01', '2026-01-01 06:00:00', 5),
    ('2026-01-01 06:30:01', '2026-01-01 07:30:00', 5),
    ('2026-01-01 08:00:01', '2026-01-01 09:00:00', 5),
    ('2026-01-01 10:00:01', '2026-01-01 11:00:00', 6),
    ('2026-01-01 11:30:01', '2026-01-01 12:30:00', 6),
    ('2026-01-01 13:00:01', '2026-01-01 14:00:00', 6),
    ('2026-01-01 15:00:01', '2026-01-01 15:40:00', 9),
    ('2026-01-01 15:56:01', '2026-01-01 16:36:00', 9),
    ('2026-01-01 16:52:01', '2026-01-01 17:32:00', 9)
]

TEST_TIME_PERIODS = [
    ('2025-04-11 09:34:01', '2025-04-11 09:44:00', 1, '冷凝盤管阻塞20%'),
    ('2025-04-11 10:06:01', '2025-04-11 10:16:00', 1, '冷凝盤管阻塞30%'),
    ('2025-04-11 10:49:01', '2025-04-11 10:59:00', 2, '蒸發盤管阻塞10%'),
    ('2025-04-11 11:21:01', '2025-04-11 11:31:00', 2, '蒸發盤管阻塞20%'),
    ('2025-04-11 11:53:01', '2025-04-11 12:03:00', 2, '蒸發盤管阻塞23%'),
    ('2025-04-11 13:05:01', '2025-04-11 13:35:00', 0, '正常資料常溫34度'),
    ('2025-04-11 14:05:01', '2025-04-11 14:15:00', 7, '蒸發風扇電流90%'),
    ('2025-04-11 14:39:01', '2025-04-11 14:49:00', 7, '蒸發風扇電流80%'),
    ('2025-04-11 15:12:01', '2025-04-11 15:22:00', 7, '蒸發風扇電流70%'),
    ('2025-04-11 16:06:01', '2025-04-11 16:16:00', 0, '正常資料高溫42.7度'),
    ('2025-04-14 09:40:01', '2025-04-14 10:00:00', 0, '正常資料低溫24度'),
    ('2025-04-14 11:30:01', '2025-04-14 11:45:00', 8, '加熱器運轉'),
    ('2025-04-14 14:05:01', '2025-04-14 14:25:00', 3, '冷媒洩漏10%'),
    ('2025-04-14 15:30:01', '2025-04-14 15:50:00', 3, '冷媒洩漏20%'),
    ('2025-12-23 11:03:01', '2025-12-23 11:23:00', 0, '正常資料低溫27度'),
    ('2025-12-23 13:40:01', '2025-12-23 13:57:00', 1, '輕度冷凝盤管阻塞'),
    ('2025-12-23 14:23:01', '2025-12-23 14:33:00', 1, '重度冷凝盤管阻塞'),
    ('2025-12-23 15:40:01', '2025-12-23 16:00:00', 3, '冷媒洩漏30%'),
    ('2026-01-01 01:00:01', '2026-01-01 01:30:00', 4, '壓縮機故障10%'),
    ('2026-01-01 02:30:01', '2026-01-01 03:00:00', 4, '壓縮機故障20%'),
    ('2026-01-01 04:00:01', '2026-01-01 04:30:00', 4, '壓縮機故障30%'),
    ('2026-01-01 06:00:01', '2026-01-01 06:30:00', 5, '冷凝風扇電流上升10%'),
    ('2026-01-01 07:30:01', '2026-01-01 08:00:00', 5, '冷凝風扇電流上升20%'),
    ('2026-01-01 09:00:01', '2026-01-01 09:30:00', 5, '冷凝風扇電流上升30%'),
    ('2026-01-01 11:00:01', '2026-01-01 11:30:00', 6, '蒸發風扇電流上升10%'),
    ('2026-01-01 12:30:01', '2026-01-01 13:00:00', 6, '蒸發風扇電流上升20%'),
    ('2026-01-01 14:00:01', '2026-01-01 14:30:00', 6, '蒸發風扇電流上升30%'),
    ('2026-01-01 15:40:01', '2026-01-01 15:56:00', 9, '加熱器故障10%'),
    ('2026-01-01 16:36:01', '2026-01-01 16:52:00', 9, '加熱器故障20%'),
    ('2026-01-01 17:32:01', '2026-01-01 17:48:00', 9, '加熱器故障30%')
]

# ==========================================
# 2. 定義輔助函數
# ==========================================
def supcon_loss(y_true, y_pred):
    """自定義之監督式對比學習損失函數"""
    temp = 1
    t = tf.cast(temp, tf.float32)
    y_true = tf.convert_to_tensor(y_true)
    y_pred = tf.convert_to_tensor(y_pred)
    
    mask = tf.math.equal(y_true, tf.transpose(y_true))
    mask = tf.cast(mask, tf.float32)
    batch_size = tf.shape(mask)[0]
    remove = tf.eye(batch_size)
    mask = mask - remove
    num_pos = tf.reduce_sum(mask, axis=1)
    
    logits = tf.matmul(y_pred, y_pred, transpose_b=True)
    logits = logits / t
    logits = (logits - tf.reduce_max(tf.stop_gradient(logits), axis=1, keepdims=True))
    exp_logits = tf.exp(logits)
    
    d_mask = tf.ones((batch_size, batch_size))
    d_mask = d_mask - remove
    denominator = exp_logits * d_mask
    denominator = tf.reduce_sum(denominator, axis=1, keepdims=True)
    
    log_probs = (logits - tf.math.log(denominator)) * mask
    log_probs = tf.reduce_sum(log_probs, axis=1)
    log_probs = tf.math.divide_no_nan(log_probs, num_pos)
    
    loss = -log_probs * t
    return loss

def create_windows(data, label, window_size, step_size):
    """將連續時間序列切割為固定長度的滑動視窗"""
    windows = []
    labels = []
    data_values = data.values if isinstance(data, pd.DataFrame) else data
    for i in range(0, len(data_values) - window_size + 1, step_size):
        windows.append(data_values[i:i + window_size])
        labels.append(label)
    return np.array(windows), np.array(labels)

def prepare_dataset(df, periods, features, window_size, step_size):
    """根據時間段過濾資料並生成三維特徵與標籤陣列"""
    x_list, y_list = [], []
    for item in periods:
        start = item[0]
        end = item[1]
        label = item[2]
        mask = (df.index >= start) & (df.index <= end)
        segment = df.loc[mask, features]
        if len(segment) >= window_size:
            x_win, y_win = create_windows(segment, label, window_size, step_size)
            x_list.append(x_win)
            y_list.append(y_win)
            
    if len(x_list) > 0:
        return np.vstack(x_list), np.concatenate(y_list)
    else:
        return np.array([]), np.array([])

def jitter_multivariate(x, sigma=0.05):
    """加入高斯雜訊 (資料增強)"""
    return x + np.random.normal(0, sigma, x.shape)

def scale_multivariate(x, sigma=0.1):
    """隨機縮放 (資料增強)"""
    factor = np.random.normal(loc=1., scale=sigma, size=(x.shape[0], 1, x.shape[2]))
    return np.multiply(x, factor)

def augmentation_multivariate(x, y, augs=['non', 'jit']):
    """整合資料增強方法產生視圖"""
    idx = np.random.permutation(len(x))
    x_shuffled = x[idx]
    y_shuffled = y[idx]
    
    aug_list_x = []
    for func in augs:
        if func == 'non':
            aug_list_x.append(x_shuffled)
        elif func == 'jit':
            aug_list_x.append(jitter_multivariate(x_shuffled))
        elif func == 'scl':
            aug_list_x.append(scale_multivariate(x_shuffled))
            
    new_x = np.concatenate(aug_list_x, axis=0)
    new_y = np.concatenate([y_shuffled] * len(augs), axis=0)
    
    idx_final = np.random.permutation(len(new_x))
    return new_x[idx_final], new_y[idx_final]

# ==========================================
# 3. 主程式執行區塊
# ==========================================
def main():
    print("正在讀取資料...")
    df = pd.read_csv(DATA_FILE_PATH)
    df['timestamp'] = pd.to_datetime(df['datetime'])
    df = df.set_index('timestamp')

    print("正在準備訓練與測試資料集...")
    x_train_raw, y_train = prepare_dataset(df, SOURCE_TIME_PERIODS, TARGET_FEATURES, WINDOW_SIZE, STEP_SIZE)
    x_test_raw, y_test = prepare_dataset(df, TEST_TIME_PERIODS, TARGET_FEATURES, WINDOW_SIZE, STEP_SIZE)

    # 展平資料以進行 MinMaxScaler 正規化
    num_train_samples, seq_len, num_features = x_train_raw.shape
    x_train_flat = x_train_raw.reshape(-1, num_features)
    
    num_test_samples = x_test_raw.shape[0]
    x_test_flat = x_test_raw.reshape(-1, num_features)

    print("進行資料正規化...")
    scaler = MinMaxScaler()
    x_train_scaled = scaler.fit_transform(x_train_flat)
    x_test_scaled = scaler.transform(x_test_flat)

    # 轉回 (樣本數, 時間步長, 特徵數) 結構
    x_train = x_train_scaled.reshape(num_train_samples, seq_len, num_features)
    x_test = x_test_scaled.reshape(num_test_samples, seq_len, num_features)

    n_class = len(np.unique(y_train))
    print(f"訓練集維度: {x_train.shape}, 測試集維度: {x_test.shape}, 總類別數: {n_class}")

    print("進行資料增強 (建立正樣本對)...")
    augs = ['non', 'jit']
    x_train_aug, y_train_aug = augmentation_multivariate(x_train, y_train, augs)
    num_views = len(augs)

    # === 模型建置與訓練 ===
    print("正在初始化 ContrastiveModel...")
    nb = 64
    input_shape = x_train.shape[1:] 

    encoder = models.cautrans_enc(
        input_shape, head_size=256, num_heads=3, num_f=256, 
        dilations=[1, 4, 16], k_size=4, dropout=0.3
    )
    projector = models.projector(
        input_shape=encoder.output_shape[1:], mlp_units=[128], mlp_dropout=0.3
    )

    model = ContrastiveModel(encoder=encoder, projector=projector)
    model.compile(e_optimizer=Adam(), e_loss_fn=supcon_loss)
    lr_schedule = ReduceLROnPlateau(monitor='e_loss', factor=0.5, patience=5, min_lr=0.00001)

    print("開始訓練 Encoder (對比學習階段)...")
    s = time.time()
    model.fit(
        x_train_aug, y_train_aug, 
        epochs=20, 
        batch_size=nb * num_views, 
        callbacks=[lr_schedule],
        verbose=1
    )
    print(f"Encoder 訓練完成，耗時: {time.time() - s:.2f} 秒")

    # === 下游分類器訓練 ===
    print("提取特徵並準備訓練分類器...")
    tr_feat = model.encoder.predict(x_train)
    te_feat = model.encoder.predict(x_test)
    y_train_squeeze = np.squeeze(y_train)

    classifier = models.MLP_cl(
        input_dim=tr_feat.shape[1], mlp_layers=[256, 64], n_class=n_class
    )
    classifier.compile(
        optimizer=Adam(learning_rate=1e-03), 
        loss='sparse_categorical_crossentropy', 
        metrics=["sparse_categorical_accuracy"]
    )
    early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

    print("開始訓練下游 MLP 分類器...")
    classifier.fit(
        tr_feat, y_train_squeeze, 
        epochs=20, 
        batch_size=64, 
        validation_split=0.1, 
        callbacks=[early_stop],
        verbose=1
    )

    # === 模型評估 ===
    print("\n========== 整體混合測試集評估 ==========")
    loss, accuracy = classifier.evaluate(te_feat, np.squeeze(y_test), verbose=0)
    print(f"整體 Accuracy: {accuracy:.4f}")

    print("\n========== 各獨立測試時間段評估結果 ==========")
    evaluation_results = []
    
    for start, end, label, desc in TEST_TIME_PERIODS:
        mask = (df.index >= start) & (df.index <= end)
        segment = df.loc[mask, TARGET_FEATURES]
        
        if len(segment) >= WINDOW_SIZE:
            x_win, y_win = create_windows(segment, label, WINDOW_SIZE, STEP_SIZE)
            
            # 使用訓練集的 scaler 進行轉換
            x_flat = x_win.reshape(-1, len(TARGET_FEATURES))
            x_scaled = scaler.transform(x_flat)
            x_eval = x_scaled.reshape(x_win.shape[0], WINDOW_SIZE, len(TARGET_FEATURES))
            
            # 提取特徵並預測
            feat_eval = model.encoder.predict(x_eval, verbose=0)
            eval_loss, eval_acc = classifier.evaluate(feat_eval, y_win, verbose=0)
            
            print(f"[{desc}] (真實類別: {label}):")
            print(f"  - 測試樣本數: {len(y_win)} 個 window")
            print(f"  - 獨立準確率: {eval_acc:.4f}\n")
            
            evaluation_results.append({
                "情境": desc,
                "類別": label,
                "樣本數": len(y_win),
                "準確率": eval_acc
            })
        else:
            print(f"[{desc}] 資料長度不足，跳過測試。\n")

    # 輸出最終表格
    df_results = pd.DataFrame(evaluation_results)
    print("獨立測試完整報告：")
    print(df_results.to_string(index=False))
    
    # 儲存 CSV (可選)
    # df_results.to_csv("test_results.csv", index=False)

if __name__ == "__main__":
    main()