# DataMining

# 🔐 Derin Öğrenme ve Makine Öğrenmesi ile Siber Güvenlik Analizi

Bu proje, küresel siber saldırı eğilimlerini anlamak, zaman serisi analizleri yapmak ve sektörel bazda saldırı tiplerini öngörmek amacıyla makine öğrenmesi ve derin öğrenme teknikleri kullanılarak geliştirilmiştir.

## 📁 Proje Yapısı

### 📊 Machine Learning Method

Bu klasörde klasik makine öğrenmesi modelleri ve sınıflandırma yöntemleri bulunmaktadır.

| Dosya Adı | Açıklama |
|-----------|----------|
| `GlobalCyberSecurityAnalysis.py` | Küresel siber güvenlik tehditleri üzerine genel analiz gerçekleştirir. |
| `NewClassfication.py` | Yeni veriler üzerinden makine öğrenmesiyle sınıflandırma işlemi yapar. |
| `TestModelClassification.py` | Eğitimli modellerin test ve doğrulama sürecini gerçekleştirir. |

---

### 📈 Financial Time Series & Trend Analizi

Bu klasör finansal kayıplar ve zaman serileri ile sektörel saldırı eğilimlerini analiz etmeye yöneliktir.

| Dosya Adı | Açıklama |
|-----------|----------|
| `FinancialTimeSeriesSector.py` | Sektörlere göre finansal kayıpların zaman serisi analizini yapar. |
| `TimeSeriesAttackTrend.py` | Saldırı trendlerini zaman içinde analiz eder. |
| `TimeSeriesLSTMCNN.py` | Zaman serisi verilerini LSTM + CNN derin öğrenme modeliyle analiz eder. |

---

### 🧠 Deep Learning

| Görsel Adı | Açıklama |
|------------|----------|
| `FinancialLossAttackTypeTargetIndustry.png` | Saldırı türlerine göre sektörlerin finansal kayıplarını gösteren grafik. |
| `FinancialLossNewNation.png` | Ülkelere göre oluşan finansal kayıpların grafiksel analizi. |

---

### 🖼️ Görseller (Images)

Bu klasörde analiz sonuçlarına ait grafik ve görseller yer almaktadır. Görseller model çıktılarını, finansal kayıpları ve saldırı tiplerini görselleştirir.

---

## 🧪 Kullanım

Projeyi kullanmadan önce gerekli Python kütüphanelerini yükleyiniz:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn keras tensorflow xgboost
