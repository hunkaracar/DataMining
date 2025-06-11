import pandas as pd
import numpy as np
# pmdarima kaldırıldı, yerine SimpleRNN eklenecek
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Conv1D, MaxPooling1D, Flatten, SimpleRNN # SimpleRNN eklendi
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

# TensorFlow ve Statsmodels uyarılarını bastırmak için
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
tf.get_logger().setLevel('ERROR')

# Statsmodels uyarıları artık ARIMA kullanılmadığı için gerekmeyebilir, ama kalsın
from statsmodels.tools.sm_exceptions import ValueWarning as sm_ValueWarning
from statsmodels.tools.sm_exceptions import ConvergenceWarning as sm_ConvergenceWarning
warnings.simplefilter('ignore', sm_ValueWarning)
warnings.simplefilter('ignore', sm_ConvergenceWarning)
warnings.simplefilter('ignore', FutureWarning)

# Veri dosyasının adını belirtin
csv_file_path = 'Global_Cybersecurity_Threats_2015-2024.csv'

# Veriyi yükleyelim
try:
    df_main = pd.read_csv(csv_file_path)
    print("Veri başarıyla yüklendi.")
    print(f"Toplam {len(df_main)} satır veri var.")
except FileNotFoundError:
    print(f"HATA: '{csv_file_path}' dosyası bulunamadı. Lütfen dosya yolunu kontrol edin.")
    exit()
except Exception as e:
    print(f"Veri yüklenirken bir hata oluştu: {e}")
    exit()

# Finansal Kayıp sütunundaki sayısal olmayan değerleri temizleme (eğer varsa)
df_main['Financial Loss (in Million $)'] = pd.to_numeric(df_main['Financial Loss (in Million $)'], errors='coerce')
df_main.dropna(subset=['Financial Loss (in Million $)'], inplace=True)


# Tahmin edilecek gelecek yıl sayısı (2025-2032)
LAST_KNOWN_YEAR = df_main['Year'].max()
FORECAST_END_YEAR = 2032
N_FORECAST_PERIODS = int(FORECAST_END_YEAR - LAST_KNOWN_YEAR)

# Derin Öğrenme Modelleri için sequence oluşturma fonksiyonu
def create_sequences_dl(data, n_lookback):
    X, y = [], []
    if len(data) <= n_lookback:
        return np.array(X), np.array(y)
    for i in range(len(data) - n_lookback):
        X.append(data[i:(i + n_lookback), 0])
        y.append(data[i + n_lookback, 0])
    X, y = np.array(X), np.array(y)
    return X, y

# Her sektör için analiz yapalım
unique_sectors = df_main['Target Industry'].unique()
all_sector_metrics = []
all_sector_forecasts_table = {}
trained_sector_models_storage = {} # Eğitilmiş model nesnelerini saklamak için EKLENDİ
all_saved_model_paths = {} # Tüm kaydedilen model yollarını saklamak için EKLENDİ

# Model dosyalarını kaydetmek için bir klasör oluşturalım (eğer yoksa) EKLENDİ
output_model_dir_sector = "saved_sector_models"
os.makedirs(output_model_dir_sector, exist_ok=True)

print(f"\nToplam {len(unique_sectors)} sektör için analiz başlatılıyor...\n")

for sector in unique_sectors:
    print(f"\n===== SEKTÖR ANALİZİ: {sector} =====")
    sector_df = df_main[df_main['Target Industry'] == sector]

    yearly_financial_loss_sector = sector_df.groupby('Year')['Financial Loss (in Million $)'].sum()

    if len(yearly_financial_loss_sector) < 5: # Zaman serisi analizi için çok kısa
        print(f"'{sector}' sektörü için yeterli yıllık veri yok (veri noktası: {len(yearly_financial_loss_sector)}). Bu sektör atlanıyor.")
        forecast_df_empty = pd.DataFrame(index=range(LAST_KNOWN_YEAR + 1, FORECAST_END_YEAR + 1))
        forecast_df_empty['SimpleRNN'] = np.nan
        forecast_df_empty['LSTM'] = np.nan
        forecast_df_empty['1D CNN'] = np.nan
        all_sector_forecasts_table[sector] = forecast_df_empty
        continue

    print(f"\n'{sector}' Sektörü - Yıllara Göre Toplam Finansal Kayıp (Milyon $):")
    print(yearly_financial_loss_sector)
    print("-" * 30)

    ts_data_sector = yearly_financial_loss_sector.copy()
    sector_safe_name = sector.replace(' ', '_').replace('/', '_').replace('&', '_') # Dosya adları için EKLENDİ
    trained_sector_models_storage[sector] = {} # Bu sektör için model saklama yeri EKLENDİ

    # --- Derin Öğrenme Modelleri için Hazırlık ---
    scaler_dl = MinMaxScaler(feature_range=(0, 1))
    scaled_data_dl = scaler_dl.fit_transform(ts_data_sector.values.reshape(-1, 1))
    n_lookback_dl = 2
    
    X_dl, y_dl_scaled = create_sequences_dl(scaled_data_dl, n_lookback_dl)

    rnn_forecast_series_sector = pd.Series(dtype='float64')
    lstm_forecast_series_sector = pd.Series(dtype='float64')
    cnn_forecast_series_sector = pd.Series(dtype='float64')
    
    rnn_mae_metric_sector, rnn_rmse_metric_sector = np.nan, np.nan
    lstm_mae_metric_sector, lstm_rmse_metric_sector = np.nan, np.nan
    cnn_mae_metric_sector, cnn_rmse_metric_sector = np.nan, np.nan
    
    model_rnn_sector_obj, model_lstm_sector_obj, model_cnn_sector_obj = None, None, None # Model nesneleri için EKLENDİ


    if X_dl.size > 0 and y_dl_scaled.size > 0:
        y_dl_original = scaler_dl.inverse_transform(y_dl_scaled.reshape(-1,1))

        # --- Basit RNN Modeli ---
        print(f"\n{sector} - Basit RNN Modeli Oluşturuluyor...")
        try:
            tf.keras.backend.clear_session() # Her model öncesi session temizliği
            X_train_rnn_sec = X_dl.reshape(X_dl.shape[0], X_dl.shape[1], 1)
            model_rnn_sector_obj = Sequential([ # Nesneyi ata
                SimpleRNN(32, activation='relu', input_shape=(n_lookback_dl, 1)),
                Dense(1)
            ], name=f"SimpleRNN_{sector_safe_name}")
            model_rnn_sector_obj.compile(optimizer='adam', loss='mse')
            model_rnn_sector_obj.fit(X_train_rnn_sec, y_dl_scaled, epochs=50, batch_size=1, verbose=0)
            trained_sector_models_storage[sector]['SimpleRNN'] = model_rnn_sector_obj # Sakla EKLENDİ

            rnn_preds_scaled_list = []
            current_batch_rnn = scaled_data_dl[-n_lookback_dl:].reshape(1, n_lookback_dl, 1)
            for _ in range(N_FORECAST_PERIODS):
                pred_rnn = model_rnn_sector_obj.predict(current_batch_rnn, verbose=0)[0]
                rnn_preds_scaled_list.append(pred_rnn)
                current_batch_rnn = np.append(current_batch_rnn[:, 1:, :], [[pred_rnn]], axis=1)
            
            rnn_fc_original = scaler_dl.inverse_transform(np.array(rnn_preds_scaled_list).reshape(-1,1))
            rnn_forecast_series_sector = pd.Series(rnn_fc_original.flatten(), index=range(LAST_KNOWN_YEAR + 1, FORECAST_END_YEAR + 1))

            train_preds_rnn_s = model_rnn_sector_obj.predict(X_train_rnn_sec, verbose=0)
            train_preds_rnn_o = scaler_dl.inverse_transform(train_preds_rnn_s)
            rnn_mae_metric_sector = mean_absolute_error(y_dl_original, train_preds_rnn_o)
            rnn_rmse_metric_sector = np.sqrt(mean_squared_error(y_dl_original, train_preds_rnn_o))
        except Exception as e_rnn:
            print(f"'{sector}' Basit RNN Hata: {e_rnn}")
        all_sector_metrics.append({'Sector': sector, 'Model': 'SimpleRNN', 'MAE': rnn_mae_metric_sector, 'RMSE': rnn_rmse_metric_sector})

        # --- LSTM Modeli ---
        print(f"\n{sector} - LSTM Modeli Oluşturuluyor...")
        try:
            tf.keras.backend.clear_session()
            X_train_lstm_sec = X_dl.reshape(X_dl.shape[0], X_dl.shape[1], 1)
            model_lstm_sector_obj = Sequential([ # Nesneyi ata
                LSTM(32, activation='relu', input_shape=(n_lookback_dl, 1)),
                Dense(1)
            ], name=f"LSTM_{sector_safe_name}")
            model_lstm_sector_obj.compile(optimizer='adam', loss='mse')
            model_lstm_sector_obj.fit(X_train_lstm_sec, y_dl_scaled, epochs=50, batch_size=1, verbose=0)
            trained_sector_models_storage[sector]['LSTM'] = model_lstm_sector_obj # Sakla EKLENDİ

            lstm_preds_scaled_list = []
            current_batch_lstm = scaled_data_dl[-n_lookback_dl:].reshape(1, n_lookback_dl, 1)
            for _ in range(N_FORECAST_PERIODS):
                pred_lstm = model_lstm_sector_obj.predict(current_batch_lstm, verbose=0)[0]
                lstm_preds_scaled_list.append(pred_lstm)
                current_batch_lstm = np.append(current_batch_lstm[:, 1:, :], [[pred_lstm]], axis=1)
            
            lstm_fc_original = scaler_dl.inverse_transform(np.array(lstm_preds_scaled_list).reshape(-1,1))
            lstm_forecast_series_sector = pd.Series(lstm_fc_original.flatten(), index=range(LAST_KNOWN_YEAR + 1, FORECAST_END_YEAR + 1))

            train_preds_lstm_s = model_lstm_sector_obj.predict(X_train_lstm_sec, verbose=0)
            train_preds_lstm_o = scaler_dl.inverse_transform(train_preds_lstm_s)
            lstm_mae_metric_sector = mean_absolute_error(y_dl_original, train_preds_lstm_o)
            lstm_rmse_metric_sector = np.sqrt(mean_squared_error(y_dl_original, train_preds_lstm_o))
        except Exception as e_lstm:
            print(f"'{sector}' LSTM Hata: {e_lstm}")
        all_sector_metrics.append({'Sector': sector, 'Model': 'LSTM', 'MAE': lstm_mae_metric_sector, 'RMSE': lstm_rmse_metric_sector})

        # --- 1D CNN Modeli ---
        print(f"\n{sector} - 1D CNN Modeli Oluşturuluyor...")
        try:
            tf.keras.backend.clear_session()
            X_train_cnn_sec = X_dl.reshape(X_dl.shape[0], X_dl.shape[1], 1)
            model_cnn_sector_obj = Sequential([ # Nesneyi ata
                Conv1D(filters=32, kernel_size=1, activation='relu', input_shape=(n_lookback_dl, 1)),
                MaxPooling1D(pool_size=1),
                Flatten(),
                Dense(16, activation='relu'),
                Dense(1)
            ], name=f"CNN_{sector_safe_name}")
            model_cnn_sector_obj.compile(optimizer='adam', loss='mse')
            model_cnn_sector_obj.fit(X_train_cnn_sec, y_dl_scaled, epochs=50, batch_size=1, verbose=0)
            trained_sector_models_storage[sector]['1D CNN'] = model_cnn_sector_obj # Sakla EKLENDİ

            cnn_preds_scaled_list = []
            current_batch_cnn = scaled_data_dl[-n_lookback_dl:].reshape(1, n_lookback_dl, 1)
            for _ in range(N_FORECAST_PERIODS):
                pred_cnn_val = model_cnn_sector_obj.predict(current_batch_cnn, verbose=0)[0] # [0] ile skaler değeri al
                cnn_preds_scaled_list.append(pred_cnn_val)
                # GÜNCELLEME: pred_cnn_val'i doğru şekilde current_batch_cnn'e ekle
                new_time_step_cnn = np.array(pred_cnn_val).reshape(1, 1, 1) # (1, 1, 1) şeklinde olmalı
                current_batch_cnn = np.append(current_batch_cnn[:, 1:, :], new_time_step_cnn, axis=1)


            cnn_fc_original = scaler_dl.inverse_transform(np.array(cnn_preds_scaled_list).reshape(-1,1))
            cnn_forecast_series_sector = pd.Series(cnn_fc_original.flatten(), index=range(LAST_KNOWN_YEAR + 1, FORECAST_END_YEAR + 1))
            
            train_preds_cnn_s = model_cnn_sector_obj.predict(X_train_cnn_sec, verbose=0)
            train_preds_cnn_o = scaler_dl.inverse_transform(train_preds_cnn_s)
            cnn_mae_metric_sector = mean_absolute_error(y_dl_original, train_preds_cnn_o)
            cnn_rmse_metric_sector = np.sqrt(mean_squared_error(y_dl_original, train_preds_cnn_o))
        except Exception as e_cnn:
            print(f"'{sector}' 1D CNN Hata: {e_cnn}")
        all_sector_metrics.append({'Sector': sector, 'Model': '1D CNN', 'MAE': cnn_mae_metric_sector, 'RMSE': cnn_rmse_metric_sector})
    else:
        print(f"'{sector}' için Derin Öğrenme sequence oluşturulamadı (veri az).")
        all_sector_metrics.append({'Sector': sector, 'Model': 'SimpleRNN', 'MAE': np.nan, 'RMSE': np.nan})
        all_sector_metrics.append({'Sector': sector, 'Model': 'LSTM', 'MAE': np.nan, 'RMSE': np.nan})
        all_sector_metrics.append({'Sector': sector, 'Model': '1D CNN', 'MAE': np.nan, 'RMSE': np.nan})


    # --- Tahminleri Kaydet (Tablo için) ---
    sector_forecast_df = pd.DataFrame(index=range(LAST_KNOWN_YEAR + 1, FORECAST_END_YEAR + 1))
    sector_forecast_df['SimpleRNN'] = rnn_forecast_series_sector.reindex(sector_forecast_df.index).round(2)
    sector_forecast_df['LSTM'] = lstm_forecast_series_sector.reindex(sector_forecast_df.index).round(2)
    sector_forecast_df['1D CNN'] = cnn_forecast_series_sector.reindex(sector_forecast_df.index).round(2)
    all_sector_forecasts_table[sector] = sector_forecast_df


    # --- Grafikleme (Her Sektör İçin) ---
    plt.figure(figsize=(14, 7))
    plt.style.use('seaborn-v0_8-darkgrid')
    plt.plot(ts_data_sector.index, ts_data_sector.values, label='Geçmiş Finansal Kayıp', color='royalblue', marker='o', linewidth=2)

    if not rnn_forecast_series_sector.empty:
        plt.plot(rnn_forecast_series_sector.index, rnn_forecast_series_sector.values, label='SimpleRNN Tahmini', color='teal', linestyle='--', marker='x')
    if not lstm_forecast_series_sector.empty:
        plt.plot(lstm_forecast_series_sector.index, lstm_forecast_series_sector.values, label='LSTM Tahmini', color='forestgreen', linestyle=':', marker='s')
    if not cnn_forecast_series_sector.empty:
        plt.plot(cnn_forecast_series_sector.index, cnn_forecast_series_sector.values, label='1D CNN Tahmini', color='purple', linestyle='-.', marker='d')

    plt.title(f"'{sector}' Sektörü Finansal Kayıp Zaman Serisi ve Tahminler (Milyon $)", fontsize=16)
    plt.xlabel("Yıl", fontsize=12)
    plt.ylabel("Finansal Kayıp (Milyon $)", fontsize=12)
    
    combined_indices_plot_sec = set(ts_data_sector.index)
    if not rnn_forecast_series_sector.empty: combined_indices_plot_sec.update(rnn_forecast_series_sector.index)
    if not lstm_forecast_series_sector.empty: combined_indices_plot_sec.update(lstm_forecast_series_sector.index)
    if not cnn_forecast_series_sector.empty: combined_indices_plot_sec.update(cnn_forecast_series_sector.index)
    if combined_indices_plot_sec:
      plt.xticks(sorted(list(combined_indices_plot_sec)), rotation=45)

    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()

    # --- Bu Sektör İçin En İyi Modeli Belirleme ve Kaydetme --- EKLENDİ
    print(f"\n--- '{sector}' Sektörü İçin En İyi Model (RMSE'ye Göre) ---")
    sector_models_perf = [m for m in all_sector_metrics if m['Sector'] == sector and pd.notnull(m['RMSE'])]
    best_model_for_this_sector_info = None
    best_rmse_for_this_sector = float('inf')

    if sector_models_perf:
        for model_perf in sector_models_perf:
            if model_perf['RMSE'] < best_rmse_for_this_sector:
                best_rmse_for_this_sector = model_perf['RMSE']
                best_model_for_this_sector_info = model_perf
    
    if best_model_for_this_sector_info:
        best_model_name = best_model_for_this_sector_info['Model']
        print(f"  En İyi Model: {best_model_name}")
        print(f"  En İyi RMSE: {best_model_for_this_sector_info['RMSE']:.2f}")
        print(f"  MAE: {best_model_for_this_sector_info['MAE']:.2f}")

        model_to_save_sector = trained_sector_models_storage.get(sector, {}).get(best_model_name)
        if model_to_save_sector:
            try:
                file_path_sector = os.path.join(output_model_dir_sector, f"best_model_{sector_safe_name}_{best_model_name.replace(' ', '')}.h5")
                model_to_save_sector.save(file_path_sector)
                print(f"  EN İYİ MODEL KAYDEDİLDİ: {file_path_sector}")
                all_saved_model_paths[f"{sector}_{best_model_name}"] = file_path_sector
            except Exception as e_save_sector:
                print(f"  HATA: '{sector}' için en iyi model ({best_model_name}) kaydedilemedi: {e_save_sector}")
        else:
            print(f"  UYARI: '{sector}' için en iyi model nesnesi ({best_model_name}) bulunamadı, kaydedilemiyor.")
    else:
        print(f"  Bu sektör için geçerli metriklerle bir model bulunamadı.")

    print(f"===== '{sector}' SEKTÖR ANALİZİ TAMAMLANDI =====")


# --- Genel Performans Metrikleri Tablosu ---
print("\n\n===== TÜM SEKTÖRLER İÇİN PERFORMANS METRİKLERİ ÖZETİ =====")
metrics_df_all_sectors = pd.DataFrame(all_sector_metrics)
if not metrics_df_all_sectors.empty:
    metrics_df_all_sectors['MAE'] = metrics_df_all_sectors['MAE'].apply(lambda x: f"{x:.2f}" if pd.notnull(x) else "-")
    metrics_df_all_sectors['RMSE'] = metrics_df_all_sectors['RMSE'].apply(lambda x: f"{x:.2f}" if pd.notnull(x) else "-")
    print(metrics_df_all_sectors.sort_values(by=['Sector', 'RMSE']).to_string(index=False)) # RMSE'ye göre sırala
else:
    print("Hesaplanacak performans metriği bulunamadı.")
print("Not: Bu metrikler modellerin geçmiş verilere ne kadar uyduğunu gösterir, gelecekteki doğrulukları değil.")


# --- Kaydedilen Tüm Model Dosyalarını Listele --- EKLENDİ
if all_saved_model_paths:
    print("\n\n===== KAYDEDİLEN TÜM MODEL DOSYALARI =====")
    for key, path in all_saved_model_paths.items():
        print(f"  {key.replace('_',' ')}: {path}")


# --- Genel Tahmin Tablosu (2026-2030) ---
print("\n\n===== TÜM SEKTÖRLER İÇİN HEDEF YIL TAHMİNLERİ (2026-2032) =====")
target_years_table_final = range(2026, FORECAST_END_YEAR + 1)

summary_table_list = []
for sector, forecast_df in all_sector_forecasts_table.items():
    row_data = {'Sector': sector}
    for year_fc in target_years_table_final:
        for model_name_fc in ['SimpleRNN', 'LSTM', '1D CNN']:
            col_name = f'{model_name_fc}_{year_fc}'
            if year_fc in forecast_df.index and model_name_fc in forecast_df.columns:
                value = forecast_df.loc[year_fc, model_name_fc]
                row_data[col_name] = f"{value:.2f}" if pd.notnull(value) else "-"
            else:
                row_data[col_name] = "-"
    summary_table_list.append(row_data)

if summary_table_list:
    summary_final_df = pd.DataFrame(summary_table_list)
    # Sütunları düzenleyelim (Sektör, sonra Model_Yıl şeklinde)
    cols_ordered = ['Sector']
    for year_fc_ord in target_years_table_final:
        for model_name_fc_ord in ['SimpleRNN', 'LSTM', '1D CNN']:
            cols_ordered.append(f'{model_name_fc_ord}_{year_fc_ord}')
    
    # Sadece var olan sütunları al
    final_cols_to_print = [col for col in cols_ordered if col in summary_final_df.columns]
    print(summary_final_df[final_cols_to_print].to_string(index=False))
else:
    print("Gelecek tahmin tablosu için veri bulunamadı.")


print("\n\nUyarı: Bu tahminler, mevcut verilere ve model varsayımlarına dayanmaktadır.")
print("Özellikle kısa zaman serilerinde yapılan uzun vadeli tahminler yüksek derecede belirsizlik içerir.")
print("Her sektör için ayrı modeller kurulmuştur ve performansları sektöre göre değişiklik gösterebilir.")