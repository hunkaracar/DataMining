import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Conv1D, MaxPooling1D, Flatten, TimeDistributed
# TimeDistributed, CNN çıktısını LSTM'in beklediği sequence formatına getirmek için kullanılabilir
# veya CNN'den sonra Flatten ve ardından LSTM için yeniden şekillendirme yapılabilir.
# Daha yaygın bir yaklaşım, CNN'i özellik çıkarıcı olarak kullanıp,
# bu özellikleri LSTM'e sequence olarak vermektir.
# Ya da Conv1D'nin çıktısını doğrudan LSTM'e uygun hale getirmek.
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import os

# TensorFlow uyarılarını bastırmak
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
tf.get_logger().setLevel('ERROR')
warnings.simplefilter('ignore', FutureWarning)


# --- Konfigürasyon ---
CSV_FILE_PATH = 'Global_Cybersecurity_Threats_2015-2024.csv'
TARGET_YEAR_PREDICTION = 2032
N_LOOKBACK = 3  # Geçmiş kaç yıla bakarak tahmin yapılacak (CNN-LSTM için biraz daha fazla olabilir)
N_FEATURES = 1  # Tek bir zaman serisi (saldırı sayısı)
# CNN-LSTM için sequence'ları biraz farklı hazırlamamız gerekebilir.
# CNN genellikle (samples, timesteps, features) alır.
# LSTM ise (samples, timesteps, features) alır.
# Conv1D'den sonra LSTM kullanacaksak, Conv1D'nin çıktısının LSTM'e uygun olması gerekir.
# Yaygın bir yapı: Input -> Conv1D -> MaxPooling1D -> LSTM -> Dense
# Bu yapıda, Conv1D ve MaxPooling1D her bir zaman adımına değil, tüm lookback penceresine uygulanır
# ve çıktısı LSTM için bir özellik vektörü haline gelir. Bu, TimeDistributed kullanmadan daha basit bir yaklaşımdır.
# Ya da her bir alt-sequence'a CNN uygulanır (TimeDistributed(Conv1D)) ve sonra LSTM.
# Şimdilik ilk yaklaşımı (Conv1D -> LSTM) deneyelim.

# Model kaydetme yolu
MODEL_SAVE_DIR = "saved_models_hybrid"
MODEL_NAME = "cnnlstm_attack_count_model.keras"
os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
MODEL_SAVE_PATH = os.path.join(MODEL_SAVE_DIR, MODEL_NAME)


# --- Veri Yükleme ve Ön İşleme ---
try:
    df = pd.read_csv(CSV_FILE_PATH)
    print("Veri başarıyla yüklendi.")
except FileNotFoundError:
    print(f"HATA: '{CSV_FILE_PATH}' dosyası bulunamadı.")
    exit()

yearly_attacks = df.groupby('Year').size()
print("\nYıllara Göre Saldırı Sayıları (Geçmiş Veri):")
print(yearly_attacks)
ts_data = yearly_attacks.copy()

if len(ts_data) <= N_LOOKBACK:
    print(f"HATA: Yeterli veri yok ({len(ts_data)}), en az {N_LOOKBACK + 1} veri noktası gerekli. N_LOOKBACK değerini azaltın.")
    exit()

# Veriyi ölçeklendirme
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(ts_data.values.reshape(-1, 1))

# Sequence oluşturma
def create_sequences(data, n_lookback):
    X, y = [], []
    for i in range(len(data) - n_lookback):
        X.append(data[i:(i + n_lookback), 0])
        y.append(data[i + n_lookback, 0])
    return np.array(X), np.array(y)

X_scaled, y_scaled = create_sequences(scaled_data, N_LOOKBACK)

if X_scaled.size == 0:
    print(f"HATA: Sequence oluşturulamadı. Veri sayısı ({len(scaled_data)}) N_LOOKBACK ({N_LOOKBACK}) için yetersiz.")
    exit()

# CNN-LSTM için girdiyi yeniden şekillendirme
# CNN katmanı için: (samples, timesteps, features)
# LSTM katmanı da Conv1D'nin çıktısını (samples, new_timesteps, filters) olarak alabilir.
# Bizim Conv1D'miz (N_LOOKBACK, 1) şeklinde bir pencere alacak.
# Çıktısı (None, N_LOOKBACK - kernel_size + 1, filters) veya pooling sonrası (None, pooled_timesteps, filters) olacak.
# Bu çıktı doğrudan LSTM'e verilebilir.
X_train_reshaped = X_scaled.reshape((X_scaled.shape[0], X_scaled.shape[1], N_FEATURES))
y_train_scaled = y_scaled # y_scaled zaten doğru formatta (samples,)

print(f"\nEğitim için sequence şekilleri: X_train_reshaped: {X_train_reshaped.shape}, y_train_scaled: {y_train_scaled.shape}")

# --- Hibrit CNN-LSTM Modeli Oluşturma ---
print("\nHibrit CNN-LSTM Modeli Oluşturuluyor...")
tf.keras.backend.clear_session() # Önceki session'ları temizle
model_cnnlstm = Sequential(name="Hybrid_CNN_LSTM")
# CNN Katmanı: Özellik çıkarımı için
model_cnnlstm.add(Conv1D(filters=64, kernel_size=2, activation='relu',
                         input_shape=(N_LOOKBACK, N_FEATURES)))
model_cnnlstm.add(MaxPooling1D(pool_size=1)) # kernel_size=2 ise pool_size=2 daha anlamlı olabilirdi ama lookback küçük.
                                           # pool_size=1 etkisizdir, kaldırılabilir veya kernel_size=1 ise kullanılabilir.
                                           # Eğer N_LOOKBACK=3, kernel_size=2 ise çıktı (None, 2, 64) olur.
                                           # Eğer N_LOOKBACK=3, kernel_size=1 ise çıktı (None, 3, 64) olur.
                                           # Şimdilik N_LOOKBACK=3, kernel_size=2 varsayalım.
# LSTM Katmanı: Zamansal bağımlılıkları öğrenmek için
# Conv1D'nin çıktısı (batch_size, steps, features) formatında olacak, bu LSTM için uygun.
model_cnnlstm.add(LSTM(units=50, activation='relu'))
model_cnnlstm.add(Dense(units=1)) # Çıkış katmanı

model_cnnlstm.compile(optimizer='adam', loss='mean_squared_error')
model_cnnlstm.summary()

# --- Modeli Eğitme ---
print("\nModel eğitiliyor...")
history = model_cnnlstm.fit(X_train_reshaped, y_train_scaled, epochs=150, batch_size=1, verbose=1)


# --- Gelecek Tahminleri ---
last_known_year = ts_data.index.max()
n_future_periods = TARGET_YEAR_PREDICTION - last_known_year

print(f"\n{n_future_periods} adım için gelecek tahmini yapılıyor ({last_known_year + 1}-{TARGET_YEAR_PREDICTION})...")

# Başlangıç için son N_LOOKBACK veriyi al
last_sequence_scaled = scaled_data[-N_LOOKBACK:]
current_batch_scaled = last_sequence_scaled.reshape((1, N_LOOKBACK, N_FEATURES))

future_predictions_scaled = []
for _ in range(n_future_periods):
    next_pred_scaled = model_cnnlstm.predict(current_batch_scaled, verbose=0)[0,0] # Çıktı (1,1) ise
    future_predictions_scaled.append(next_pred_scaled)
    # Tahmini yeni zaman adımı olarak ekle ve en eskiyi at
    new_time_step = np.array([[[next_pred_scaled]]]) # (1,1,1) şeklinde
    current_batch_scaled = np.append(current_batch_scaled[:, 1:, :], new_time_step, axis=1)

# Tahminleri orijinal ölçeğe döndürme
future_predictions_original = scaler.inverse_transform(np.array(future_predictions_scaled).reshape(-1, 1))

# Tahminler için Pandas Serisi oluşturma
future_years_index = pd.RangeIndex(start=last_known_year + 1,
                                   stop=last_known_year + 1 + n_future_periods)
predictions_series = pd.Series(future_predictions_original.flatten(), index=future_years_index)

print("\nCNN-LSTM Tahminleri:")
print(predictions_series.round(0))


# --- Performans Metrikleri (Eğitim Verisi Üzerinde) ---
print("\nPerformans Metrikleri (Geçmiş Verilere Uyum):")
train_predictions_scaled = model_cnnlstm.predict(X_train_reshaped, verbose=0)
train_predictions_original = scaler.inverse_transform(train_predictions_scaled)
y_train_original = scaler.inverse_transform(y_train_scaled.reshape(-1,1))

if len(y_train_original) > 0:
    mae = mean_absolute_error(y_train_original, train_predictions_original)
    rmse = np.sqrt(mean_squared_error(y_train_original, train_predictions_original))
    print(f"  MAE (eğitim): {mae:.2f}")
    print(f"  RMSE (eğitim): {rmse:.2f}")
else:
    print("  Eğitim verisi metrikleri hesaplanamadı.")
print("Not: Bu metrikler modelin geçmiş verilere ne kadar uyduğunu gösterir, gelecekteki doğrulukları değil.")


# --- Modeli Kaydetme ---
print(f"\nModel şuraya kaydediliyor: {MODEL_SAVE_PATH}")
try:
    model_cnnlstm.save(MODEL_SAVE_PATH)
    print("Model başarıyla kaydedildi.")
except Exception as e:
    print(f"Model kaydedilirken hata oluştu: {e}")


# --- Zaman Serisi Grafiği ---
plt.figure(figsize=(16, 9))
plt.style.use('seaborn-v0_8-darkgrid')

plt.plot(ts_data.index, ts_data.values, label='Geçmiş Saldırı Sayısı', color='royalblue', marker='o', linewidth=2)
plt.plot(predictions_series.index, predictions_series.values,
         label=f'CNN-LSTM Tahmini ({future_years_index.min()}-{future_years_index.max()})',
         color='darkorange', linestyle='--', marker='s')

plt.title(f'Yıllara Göre Siber Saldırı Sayısı ve CNN-LSTM Tahminleri (->{TARGET_YEAR_PREDICTION})', fontsize=18, fontweight='bold')
plt.xlabel('Yıl', fontsize=15)
plt.ylabel('Saldırı Sayısı', fontsize=15)

all_indices = sorted(list(set(ts_data.index) | set(predictions_series.index)))
plt.xticks(all_indices, rotation=45, ha="right")
plt.yticks(fontsize=10)
plt.legend(fontsize=12, loc='best')
plt.grid(True, which='major', linestyle='--', linewidth=0.7)
plt.tight_layout()
sns.despine()
plt.show()


print("\n\nUyarı: Bu tahminler, mevcut verilere ve model varsayımlarına dayanmaktadır.")
print("Özellikle kısa zaman serilerinde (10-15 yıllık veri gibi) yapılan uzun vadeli tahminler")
print("yüksek derecede belirsizlik içerir ve dikkatle yorumlanmalıdır.")
print(f"Eğitilmiş model '{MODEL_NAME}' adıyla '{MODEL_SAVE_DIR}' klasörüne kaydedilmiştir.")