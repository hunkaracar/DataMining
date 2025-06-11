import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Conv1D, MaxPooling1D, Flatten
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.losses import Huber
from prophet import Prophet
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings
import joblib # MODEL KAYDETMEK İÇİN EKLENDİ

from tensorflow.keras import backend as K

warnings.filterwarnings('ignore')
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
tf.get_logger().setLevel('ERROR')
import logging
logging.getLogger('prophet').setLevel(logging.ERROR)
logging.getLogger('cmdstanpy').setLevel(logging.ERROR)

def find_best_arima_order(data, max_p=2, max_d=1, max_q=2):
    best_aic = float('inf')
    best_order = (1, 1, 1) # Varsayılan bir order
    if len(data) < 5: # Çok az veri varsa basit bir order dene veya atla
        print("ARIMA order bulmak için veri çok az, varsayılan (1,1,1) deneniyor veya basit bir model kullanılabilir.")
        # return (1,0,0) # veya None
    for p in range(max_p + 1):
        for d in range(max_d + 1):
            for q in range(max_q + 1):
                if p == 0 and d == 0 and q == 0: # Trivial model atla
                    continue
                try:
                    model = ARIMA(data, order=(p, d, q))
                    results = model.fit()
                    if results.aic < best_aic:
                        best_aic = results.aic
                        best_order = (p, d, q)
                except Exception as e_find_order:
                    # print(f"Order ({p},{d},{q}) denenirken hata: {e_find_order}") # Detaylı hata logu için açılabilir
                    continue
    print(f"Bulunan en iyi ARIMA order: {best_order} (AIC: {best_aic if best_aic != float('inf') else 'N/A'})")
    return best_order

def preprocess_data_for_dl(data_series):
    data_values = data_series.values.astype(float)
    if len(data_values) < 2: # Ölçekleyici için en az 2 veri noktası
        print("UYARI: Ölçekleme için yetersiz veri. Ölçekleyici uygulanamıyor.")
        return data_values.reshape(-1,1), None # Scaler None döner
    Q1 = np.percentile(data_values, 25)
    Q3 = np.percentile(data_values, 75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    data_cleaned = np.clip(data_values, lower_bound, upper_bound)
    scaler = MinMaxScaler(feature_range=(0, 1))
    data_scaled = scaler.fit_transform(data_cleaned.reshape(-1, 1))
    return data_scaled, scaler

def create_sequences(data_scaled, seq_length):
    X, y = [], []
    if data_scaled is None or len(data_scaled) <= seq_length:
        return np.array(X), np.array(y)
    for i in range(len(data_scaled) - seq_length):
        X.append(data_scaled[i:(i + seq_length), 0])
        y.append(data_scaled[i + seq_length, 0])
    return np.array(X), np.array(y)

def create_cnn_model(seq_length, filters=32, kernel_s=1, pool_s=1, dense_units=16, model_name="cnn_model"):
    K.clear_session()
    model = Sequential(name=model_name)
    model.add(Conv1D(filters=filters, kernel_size=kernel_s, activation='relu', input_shape=(seq_length, 1)))
    model.add(MaxPooling1D(pool_size=pool_s))
    model.add(Flatten())
    model.add(Dense(dense_units, activation='relu'))
    model.add(Dense(1))
    model.build(input_shape=(None, seq_length, 1))
    model.compile(optimizer='adam', loss=Huber(), metrics=['mae'])
    return model

def calculate_metrics(y_true, y_pred):
    y_true = np.array(y_true).flatten()
    y_pred = np.array(y_pred).flatten()
    min_len = min(len(y_true), len(y_pred))
    if min_len == 0:
        return {'MSE': np.nan, 'RMSE': np.nan, 'MAE': np.nan, 'R2': np.nan}
    y_true = y_true[:min_len]
    y_pred = y_pred[:min_len]
    try:
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
    except ValueError: # Eğer y_true ve y_pred tamamen farklıysa veya boşsa
        return {'MSE': np.nan, 'RMSE': np.nan, 'MAE': np.nan, 'R2': np.nan}
    return {'MSE': mse, 'RMSE': rmse, 'MAE': mae, 'R2': r2}

def iterative_forecast_cnn(model, initial_sequence_scaled, n_steps, scaler, seq_length):
    if initial_sequence_scaled is None or initial_sequence_scaled.size == 0 or initial_sequence_scaled.shape[0] != 1 or initial_sequence_scaled.shape[1] != seq_length:
        print(f"iterative_forecast_cnn: Başlangıç sequence'ı geçersiz. Shape: {initial_sequence_scaled.shape if initial_sequence_scaled is not None else 'None'}, Beklenen (1, {seq_length}, 1)")
        return np.array([])

    current_sequence_np = np.copy(initial_sequence_scaled) # Zaten (1, seq_length, 1) olmalı
    forecast_scaled_list = []
    
    for i_iter in range(n_steps):
        try:
            pred_scaled_raw = model.predict(current_sequence_np, verbose=0)
            pred_scaled_value = pred_scaled_raw[0,0] # Model çıktısı (1,1) ise
            forecast_scaled_list.append(pred_scaled_value)
            new_time_step = np.array([[[pred_scaled_value]]]) # (1,1,1) şeklinde
            current_sequence_np = np.concatenate((current_sequence_np[:, 1:, :], new_time_step), axis=1)
        except Exception as e_pred:
            print(f"iterative_forecast_cnn içinde model.predict hatası (adım {i_iter+1}, model: {model.name}): {e_pred}")
            print(f"Hata anındaki input_sequence_np şekli: {current_sequence_np.shape if current_sequence_np is not None else 'None'}")
            if forecast_scaled_list and scaler is not None: return scaler.inverse_transform(np.array(forecast_scaled_list).reshape(-1, 1)).flatten()
            return np.array([])
            
    if not forecast_scaled_list: return np.array([])
    if scaler is None: # Eğer scaler yoksa ölçeklenmiş değerleri döndür (ön işleme yapılamadıysa)
        print("UYARI: iterative_forecast_cnn'de scaler None, ölçeklenmiş değerler dönülüyor.")
        return np.array(forecast_scaled_list).flatten()
    return scaler.inverse_transform(np.array(forecast_scaled_list).reshape(-1, 1)).flatten()

# --- Ana İşlem Bloğu ---
df = pd.read_csv('Global_Cybersecurity_Threats_2015-2024.csv')
attacks_by_year_type = df.groupby(['Year', 'Attack Type']).size().reset_index(name='count')
attacks_pivot = attacks_by_year_type.pivot(index='Year', columns='Attack Type', values='count').fillna(0)

all_results = {}
trained_models_storage = {}

num_attack_types = len(attacks_pivot.columns)
cols_subplot = 2
rows_subplot = (num_attack_types + cols_subplot - 1) // cols_subplot
plt.figure(figsize=(8 * cols_subplot, 6 * rows_subplot))

LAST_KNOWN_YEAR_DATA = attacks_pivot.index.max()
FORECAST_START_YEAR = LAST_KNOWN_YEAR_DATA + 1
FORECAST_END_YEAR_TARGET = 2032
N_FUTURE_STEPS = FORECAST_END_YEAR_TARGET - LAST_KNOWN_YEAR_DATA
future_years_index = pd.RangeIndex(start=FORECAST_START_YEAR, stop=FORECAST_END_YEAR_TARGET + 1)

for i, attack_type in enumerate(attacks_pivot.columns):
    print(f"\n--- Saldırı Türü Analizi: {attack_type} ---")
    
    ts_data_original = attacks_pivot[attack_type].copy()
    if len(ts_data_original) < 3 : # ARIMA ve diğer modeller için çok az veri
        print(f"'{attack_type}' için veri sayısı ({len(ts_data_original)}) analiz için yetersiz. Atlanıyor.")
        all_results[attack_type] = { # Boş sonuçlar ekle
            'ARIMA_metrics': {}, 'Prophet_metrics': {}, 'CNN_metrics': {},
            'ARIMA_forecast': pd.Series(index=future_years_index, dtype=float),
            'Prophet_forecast': pd.Series(index=future_years_index, dtype=float),
            'CNN_forecast': pd.Series(index=future_years_index, dtype=float)
        }
        trained_models_storage[attack_type] = {}
        continue

    attack_type_safe_name = attack_type.replace(' ', '_').replace('/', '_')
    trained_models_storage[attack_type] = {}
    
    # --- ARIMA Modeli ---
    print("ARIMA Modeli işleniyor...")
    arima_pred_in_sample = np.array([])
    arima_forecast_future = np.array([])
    best_order_arima = (np.nan, np.nan, np.nan)
    arima_model_fitted_obj = None
    try:
        best_order_arima = find_best_arima_order(ts_data_original.values)
        if best_order_arima: # Eğer geçerli bir order bulunduysa
            arima_model_fitted_obj = ARIMA(ts_data_original.values, order=best_order_arima).fit()
            trained_models_storage[attack_type]['ARIMA'] = arima_model_fitted_obj
            arima_pred_in_sample = arima_model_fitted_obj.predict(start=0, end=len(ts_data_original)-1)
            arima_forecast_future = arima_model_fitted_obj.predict(start=len(ts_data_original), 
                                                               end=len(ts_data_original) + N_FUTURE_STEPS - 1)
        else:
            print(f"ARIMA için uygun order bulunamadı ({attack_type}).")
    except Exception as e_arima:
        print(f"ARIMA hatası ({attack_type}): {e_arima}")

    # --- Prophet Modeli ---
    print("Prophet Modeli işleniyor...")
    prophet_pred_in_sample = np.array([])
    prophet_forecast_future = np.array([])
    prophet_model_obj = None
    try:
        prophet_df_train = pd.DataFrame({'ds': pd.to_datetime(ts_data_original.index, format='%Y'), 
                                         'y': ts_data_original.values})
        prophet_model_obj = Prophet(growth='linear', yearly_seasonality=False, weekly_seasonality=False, daily_seasonality=False)
        prophet_model_obj.add_seasonality(name='custom_yearly', period=365.25, fourier_order=1) # Yıllık periyot
        prophet_model_obj.fit(prophet_df_train)
        trained_models_storage[attack_type]['Prophet'] = prophet_model_obj
        prophet_pred_in_sample_df = prophet_model_obj.predict(prophet_df_train[['ds']])
        prophet_pred_in_sample = prophet_pred_in_sample_df['yhat'].values
        future_df_prophet = prophet_model_obj.make_future_dataframe(periods=N_FUTURE_STEPS, freq='A-DEC') # Yıllık sonu
        prophet_forecast_future_df = prophet_model_obj.predict(future_df_prophet)
        prophet_forecast_future = prophet_forecast_future_df['yhat'][-N_FUTURE_STEPS:].values
    except Exception as e_prophet:
        print(f"Prophet hatası ({attack_type}): {e_prophet}")

    # --- 1D CNN Modeli ---
    # (Bu model artık kaydedilmeyecek ama metrikler için eğitilebilir)
    print("1D CNN Modeli işleniyor (sadece metrikler için)...")
    cnn_pred_in_sample_orig = np.array([])
    # cnn_forecast_future = np.array([]) # Gelecek tahmini yapılmayacaksa bu satıra gerek yok
    cnn_model_dl_obj = None
    data_scaled_dl, scaler_dl = preprocess_data_for_dl(ts_data_original)
    seq_length_dl = 2
    
    if scaler_dl is not None: # Scaler başarıyla oluşturulduysa devam et
        X_dl_seq, y_dl_seq = create_sequences(data_scaled_dl, seq_length_dl)
        early_stopping_cb = EarlyStopping(monitor='loss', patience=10, restore_best_weights=True, verbose=0) # monitor='loss'

        if X_dl_seq.size > 0 and y_dl_seq.size > 0:
            try:
                cnn_model_dl_obj = create_cnn_model(seq_length_dl, model_name=f"cnn_{attack_type_safe_name}")
                
                current_batch_size = 4 # Daha küçük batch size
                if len(X_dl_seq) >= current_batch_size :
                     cnn_model_dl_obj.fit(X_dl_seq, y_dl_seq, epochs=100, batch_size=current_batch_size, # Daha az epoch
                                     callbacks=[early_stopping_cb], verbose=0) # Validation_split kaldırıldı
                else: 
                     print(f"{attack_type} için CNN modeli batch_size yetersiz olduğundan farklı batch_size ile eğitiliyor.")
                     adjusted_batch_size = max(1, len(X_dl_seq))
                     cnn_model_dl_obj.fit(X_dl_seq, y_dl_seq, epochs=100, batch_size=adjusted_batch_size,
                                     callbacks=[early_stopping_cb], verbose=0)

                # trained_models_storage[attack_type]['1D CNN'] = cnn_model_dl_obj # Artık kaydetmiyoruz
                cnn_pred_in_sample_scaled = cnn_model_dl_obj.predict(X_dl_seq, verbose=0)
                cnn_pred_in_sample_orig = scaler_dl.inverse_transform(cnn_pred_in_sample_scaled).flatten()
                
                # Gelecek tahmini yapmayacağız, bu yüzden aşağıdaki kısım yorum satırı
                # if len(data_scaled_dl) >= seq_length_dl:
                #     last_sequence_for_cnn = data_scaled_dl[-seq_length_dl:].reshape(1, seq_length_dl, 1)
                #     cnn_forecast_future = iterative_forecast_cnn(cnn_model_dl_obj, last_sequence_for_cnn, N_FUTURE_STEPS, scaler_dl, seq_length_dl)
            except Exception as e_cnn:
                print(f"CNN hatası ({attack_type}): {e_cnn}")
        else:
            print(f"{attack_type} için 1D CNN modeli sequence oluşturulamadığı için atlandı.")
    else:
        print(f"{attack_type} için 1D CNN modeli (scaler oluşturulamadığı için) atlandı.")


    # --- Metrikleri Hesaplama ---
    actual_for_arima_metrics = ts_data_original.values
    actual_for_prophet_metrics = ts_data_original.values
    actual_for_cnn_metrics = ts_data_original.values[seq_length_dl:] if X_dl_seq.size > 0 and scaler_dl is not None else np.array([])
    
    all_results[attack_type] = {
        'ARIMA_order': best_order_arima,
        'ARIMA_metrics': calculate_metrics(actual_for_arima_metrics, arima_pred_in_sample) if arima_pred_in_sample.size > 0 else {},
        'Prophet_metrics': calculate_metrics(actual_for_prophet_metrics, prophet_pred_in_sample) if prophet_pred_in_sample.size > 0 else {},
        'CNN_metrics': calculate_metrics(actual_for_cnn_metrics, cnn_pred_in_sample_orig) if cnn_pred_in_sample_orig.size > 0 else {}
    }
    
    all_results[attack_type]['ARIMA_forecast'] = pd.Series(arima_forecast_future, index=future_years_index) if arima_forecast_future.size > 0 else pd.Series(index=future_years_index, dtype=float)
    all_results[attack_type]['Prophet_forecast'] = pd.Series(prophet_forecast_future, index=future_years_index) if prophet_forecast_future.size > 0 else pd.Series(index=future_years_index, dtype=float)
    # CNN için gelecek tahmini serisi artık doldurulmuyor
    all_results[attack_type]['CNN_forecast'] = pd.Series(index=future_years_index, dtype=float)


    # --- Görselleştirme ---
    ax = plt.subplot(rows_subplot, cols_subplot, i + 1)
    ax.plot(ts_data_original.index, ts_data_original.values, label='Gerçek Veri', marker='o', color='royalblue', linewidth=2)
    
    if arima_pred_in_sample.size > 0:
        ax.plot(ts_data_original.index, arima_pred_in_sample, label='ARIMA (Geçmiş)', linestyle='--', color='darkorange')
    if arima_forecast_future.size > 0:
        ax.plot(future_years_index, arima_forecast_future, label='ARIMA (Gelecek)', linestyle='--', color='darkorange', marker='^')
    
    if prophet_pred_in_sample.size > 0:
        ax.plot(ts_data_original.index, prophet_pred_in_sample, label='Prophet (Geçmiş)', linestyle=':', color='mediumseagreen')
    if prophet_forecast_future.size > 0:
        ax.plot(future_years_index, prophet_forecast_future, label='Prophet (Gelecek)', linestyle=':', color='mediumseagreen', marker='p')
        
    # CNN sadece geçmiş uyum için çizilebilir, gelecek tahmini yok
    if cnn_pred_in_sample_orig.size > 0 and len(actual_for_cnn_metrics) == len(cnn_pred_in_sample_orig) and len(actual_for_cnn_metrics) > 0 :
        # CNN tahminleri, sequence oluşturma nedeniyle orijinal verinin başını kaçırır.
        # Bu yüzden x eksenini ona göre ayarlamalıyız.
        cnn_x_axis = ts_data_original.index[len(ts_data_original) - len(cnn_pred_in_sample_orig):]
        if len(cnn_x_axis) == len(cnn_pred_in_sample_orig): # Uzunluklar eşleşiyorsa çiz
             ax.plot(cnn_x_axis, cnn_pred_in_sample_orig, label='1D CNN (Geçmiş Uyum)', linestyle='-.', color='purple')
        
    ax.set_title(f'{attack_type} Saldırı Trendi', fontsize=12)
    ax.set_xlabel('Yıl', fontsize=10)
    ax.set_ylabel('Saldırı Sayısı', fontsize=10)
    ax.legend(fontsize=8, loc='best')
    ax.grid(True, linestyle='--', alpha=0.7)
    
    combined_x_ticks = sorted(list(set(ts_data_original.index) | set(future_years_index)))
    ax.set_xticks(combined_x_ticks)
    ax.tick_params(axis='x', rotation=45, labelsize=8)
    ax.tick_params(axis='y', labelsize=8)

plt.tight_layout(pad=3.0, h_pad=3.0, w_pad=2.0)
plt.suptitle("Saldırı Türlerine Göre Yıllık Trendler ve Gelecek Tahminleri", fontsize=16, fontweight='bold', y=1.00)
plt.show()

# --- Performans Metriklerini Yazdırma ---
print("\n===== GEÇMİŞ VERİLERE UYUM PERFORMANS METRİKLERİ =====")
for attack_type, res in all_results.items():
    print(f"\n--- Saldırı Türü: {attack_type} ---")
    print(f"  ARIMA (Order: {res.get('ARIMA_order', 'N/A')}):")
    metrics_arima = res.get('ARIMA_metrics', {})
    if metrics_arima and not all(np.isnan(list(metrics_arima.values()))):
        for metric, value in metrics_arima.items(): print(f"    {metric}: {value:.4f}")
    else: print("    Metrikler hesaplanamadı veya NaN.")

    print(f"  Prophet:")
    metrics_prophet = res.get('Prophet_metrics', {})
    if metrics_prophet and not all(np.isnan(list(metrics_prophet.values()))):
        for metric, value in metrics_prophet.items(): print(f"    {metric}: {value:.4f}")
    else: print("    Metrikler hesaplanamadı veya NaN.")
        
    print(f"  1D CNN (Sadece Geçmiş Uyum):") # CNN artık kaydedilmiyor, sadece geçmiş uyum
    metrics_cnn = res.get('CNN_metrics', {})
    if metrics_cnn and not all(np.isnan(list(metrics_cnn.values()))):
        for metric, value in metrics_cnn.items(): print(f"    {metric}: {value:.4f}")
    else: print("    Metrikler hesaplanamadı veya NaN.")


# --- BELİRLENEN MODELLERİN KAYDEDİLMESİ (İSTEĞİNİZE GÖRE GÜNCELLENDİ) ---
print("\n\n===== HER SALDIRI TÜRÜ İÇİN BELİRLENEN MODELİN KAYDEDİLMESİ =====")
saved_model_paths = {} 

output_model_dir = "TimeSeriesAttackTrend_Models" # Kaydedilecek klasör adı
os.makedirs(output_model_dir, exist_ok=True)

attack_type_to_model_map = {
    "SQL Injection": "ARIMA",
    "Phishing": "ARIMA",
    "Man-in-the-Middle": "ARIMA",
    "Malware": "ARIMA",
    "DDoS": "Prophet",
    "Ransomware": "Prophet"
}

for attack_type, res_data in all_results.items(): # res_data olarak değiştirdim çakışmasın diye
    chosen_model_name = attack_type_to_model_map.get(attack_type)
    
    if not chosen_model_name:
        print(f"\n--- Saldırı Türü: {attack_type} ---")
        print(f"  UYARI: Bu saldırı türü ({attack_type}) için önceden tanımlanmış bir model (ARIMA/Prophet) yok. Atlanıyor.")
        continue

    print(f"\n--- Saldırı Türü: {attack_type} (Kaydedilecek Model: {chosen_model_name}) ---")
    
    attack_type_safe_name = attack_type.replace(' ', '_').replace('/', '_')
    model_to_save = None
    model_metrics_to_report = {}
    model_details_for_print = ""

    if chosen_model_name == "ARIMA":
        model_to_save = trained_models_storage.get(attack_type, {}).get("ARIMA")
        model_metrics_to_report = res_data.get('ARIMA_metrics', {})
        model_details_for_print = f"ARIMA (Order: {res_data.get('ARIMA_order', 'N/A')})"
    elif chosen_model_name == "Prophet":
        model_to_save = trained_models_storage.get(attack_type, {}).get("Prophet")
        model_metrics_to_report = res_data.get('Prophet_metrics', {})
        model_details_for_print = "Prophet"
    
    if model_to_save: # Sadece model nesnesi varsa devam et
        print(f"  Model Detayları: {model_details_for_print}")
        if model_metrics_to_report: # Metrikler varsa yazdır
            current_rmse_report = model_metrics_to_report.get('RMSE', float('inf'))
            if pd.notnull(current_rmse_report):
                print(f"  RMSE: {current_rmse_report:.4f}")
                # Diğer metrikler zaten yukarıda genel metrikler bölümünde yazdırılıyor.
                # İstenirse burada da spesifik olarak yazdırılabilir.
            else:
                print("  UYARI: Bu model için RMSE hesaplanamamış veya NaN.")
        else:
            print("  UYARI: Bu model için metrik bilgisi bulunamadı.")


        # Model Kaydetme
        try:
            file_path = ""
            if chosen_model_name == "ARIMA":
                file_path = os.path.join(output_model_dir, f"best_ARIMA_model_{attack_type_safe_name}.pkl")
                joblib.dump(model_to_save, file_path)
            elif chosen_model_name == "Prophet":
                file_path = os.path.join(output_model_dir, f"best_prophet_model_{attack_type_safe_name}.pkl")
                joblib.dump(model_to_save, file_path)
            
            if file_path:
                print(f"  MODEL KAYDEDİLDİ: {file_path}")
                saved_model_paths[f"{attack_type}_{chosen_model_name}"] = file_path
        except Exception as e_save:
            print(f"  HATA: Model ({chosen_model_name} for {attack_type}) kaydedilemedi: {e_save}")
    else:
        print(f"  UYARI: Seçilen model nesnesi ({chosen_model_name} for {attack_type}) bulunamadı, kaydedilemiyor.")


if saved_model_paths:
    print("\n\n--- KAYDEDİLEN MODEL DOSYALARI (TimeSeriesAttackTrend) ---")
    for key, path in saved_model_paths.items():
        print(f"  {key}: {path}")


# --- Gelecek Tahminlerini Tablo Halinde Yazdırma (2026-FORECAST_END_YEAR_TARGET) ---
print("\n\n===== GELECEK TAHMİNLERİ =====")
target_forecast_years = range(2026, FORECAST_END_YEAR_TARGET + 1)

summary_forecast_df_list = []
for attack_type, res_data_fc in all_results.items(): # res_data_fc
    # Eğer bu attack_type için model kaydedilmediyse (attack_type_to_model_map'te yoksa) atla
    if attack_type not in attack_type_to_model_map:
        continue

    row = {'Attack Type': attack_type}
    # Sadece kaydedilen modelin tahminini tabloya ekle
    chosen_model_for_fc = attack_type_to_model_map.get(attack_type)
    if chosen_model_for_fc == "ARIMA":
        model_name_key_fc = 'ARIMA_forecast'
    elif chosen_model_for_fc == "Prophet":
        model_name_key_fc = 'Prophet_forecast'
    else: # Beklenmedik bir durum, bu saldırı türü için hangi modelin tahmini alınacak?
        continue # Veya hata logla

    model_short_name_fc = chosen_model_for_fc
    forecast_series_fc = res_data_fc.get(model_name_key_fc, pd.Series(dtype=float))
    
    for year in target_forecast_years:
        value = forecast_series_fc.get(year, np.nan)
        row[f'{model_short_name_fc} {year}'] = f"{value:.0f}" if pd.notnull(value) else "-"
    summary_forecast_df_list.append(row)

if summary_forecast_df_list:
    summary_forecast_df = pd.DataFrame(summary_forecast_df_list)
    if not summary_forecast_df.empty:
        # Sütunları dinamik olarak oluştur, sadece dolu olanları
        cols_order_fc = ['Attack Type']
        # DataFrame'deki mevcut sütunlardan yıl ve model adlarını çıkar
        present_model_year_cols = [col for col in summary_forecast_df.columns if col != 'Attack Type']
        # Bu sütunları sırala (isteğe bağlı, ama daha düzenli olur)
        # Önce yıla, sonra modele göre sıralayabiliriz veya tam tersi
        # Şimdilik bulduğu sırada bırakalım veya basitçe ekleyelim
        cols_order_fc.extend(sorted(present_model_year_cols)) # Alfabetik sıralama

        final_cols_fc = [col for col in cols_order_fc if col in summary_forecast_df.columns]
        print(summary_forecast_df[final_cols_fc].to_string(index=False))
    else:
        print("Gelecek tahminleri oluşturulamadı (DataFrame boş).")
else:
    print("Gelecek tahminleri için veri listesi boş.")


print("\n\nUyarı: Bu tahminler, mevcut verilere ve model varsayımlarına dayanmaktadır.")
print("Özellikle kısa zaman serilerinde yapılan uzun vadeli tahminler yüksek derecede belirsizlik içerir.")