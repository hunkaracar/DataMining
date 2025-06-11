import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_predict, StratifiedKFold, learning_curve
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import seaborn as sns
# LogisticRegression ve MLPClassifier gibi diğer modelleri isterseniz aktif edebilirsiniz.
# from sklearn.linear_model import LogisticRegression
# from sklearn.neural_network import MLPClassifier

# 1. 📁 Veri setinin yolu
file_path = 'Global_Cybersecurity_Threats_2015-2024.csv'

# 2. 📊 Veri setini oku
try:
    df = pd.read_csv(file_path, encoding='utf-8') # Farklı encoding'leri deneyebilirsiniz: 'latin1', 'iso-8859-1'
except FileNotFoundError:
    print(f"Hata: '{file_path}' dosyası bulunamadı.")
    exit()
except Exception as e:
    print(f"CSV okunurken hata oluştu: {e}")
    exit()

# 3. 🎯 Hedef değişken ve Özelliklerin Belirlenmesi
target_column = 'Target Industry' # Tahmin edilecek sütun
# target_column = 'Attack Type'
# target_column = 'Security Vulnerability Type'

if target_column not in df.columns:
    print(f"Hata: Hedef sütun '{target_column}' veri setinde bulunamadı.")
    print(f"Mevcut sütunlar: {df.columns.tolist()}")
    exit()

y = df[target_column]
print(f"Hedef Değişken: {target_column}")
print(f"Sınıf Sayısı: {y.nunique()}")
print(f"Sınıf Dağılımı:\n{y.value_counts(normalize=True)}\n")


# Özellik matrisi (X) için kullanılmayacak sütunlar
# 'Financial Loss (in Million $)' sütunu genellikle bir sonuçtur ve sızıntıya neden olabilir.
# Eğer böyle bir sütun yoksa veya analizde kullanılacaksa, bu listeyi güncelleyin.
columns_to_drop_for_X = [target_column]
# 'Financial Loss (in Million $)' sütununun varlığını kontrol edelim
financial_loss_col = 'Financial Loss (in Million $)' # CSV'deki tam adı bu olmayabilir, kontrol edin
if financial_loss_col in df.columns:
    columns_to_drop_for_X.append(financial_loss_col)
else:
    print(f"Uyarı: '{financial_loss_col}' sütunu veri setinde bulunamadı ve X'ten çıkarılmayacak.")


X_raw = df.drop(columns=columns_to_drop_for_X, errors='ignore')

# 4. 🔢 Kategorik verileri sayısala çevir (One-Hot Encoding)
# Sadece object veya category tipindeki sütunları one-hot encode et
categorical_cols = X_raw.select_dtypes(include=['object', 'category']).columns
X_encoded = pd.get_dummies(X_raw, columns=categorical_cols, dummy_na=False) # dummy_na=False eksik değerler için ayrı sütun oluşturmaz
print(f"Özellik matrisi X'in şekli (kodlama sonrası): {X_encoded.shape}")

# 5. 划分 Veriyi Eğitim ve Test Setlerine Ayır
# Stratify=y, eğitim ve test setlerindeki sınıf oranlarını korur
X_train_raw, X_test_raw, y_train, y_test = train_test_split(
    X_encoded, y, test_size=0.20, random_state=42, stratify=y
)
print(f"Eğitim seti boyutu: {X_train_raw.shape}, Test seti boyutu: {X_test_raw.shape}")

# 6. 🔧 Veriyi ölçeklendir (standartlaştır)
# Scaler'ı SADECE eğitim verisi üzerinde fit et, sonra hem eğitim hem de test verisini transform et
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_raw)
X_test_scaled = scaler.transform(X_test_raw)


# 7. 🔁 Stratified K-Fold çapraz doğrulama (Eğitim verisi üzerinde kullanılacak)
kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# 8. 🤖 Kullanılacak modeller
models = {
    "KNN (k=5)": KNeighborsClassifier(n_neighbors=5),
    "SVC (linear)": SVC(kernel='linear', probability=True, random_state=42), # random_state eklendi
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
}

# 9. 📈 Çapraz Doğrulama Sonuçlarını kaydetmek için liste
cv_results = []

# 10. 🔁 Her model için Çapraz Doğrulama ile Değerlendirme (Eğitim Verisi Üzerinde)
print("\n--- Çapraz Doğrulama Sonuçları (Eğitim Verisi Üzerinde) ---")
for name, model in models.items():
    print(f"\nModel: {name}")

    # K-Fold doğrulama tahminleri (eğitim verisi üzerinde)
    y_pred_cv = cross_val_predict(model, X_train_scaled, y_train, cv=kfold)

    # Değerlendirme metrikleri
    acc = accuracy_score(y_train, y_pred_cv)
    prec = precision_score(y_train, y_pred_cv, average='weighted', zero_division=0)
    rec = recall_score(y_train, y_pred_cv, average='weighted', zero_division=0)
    f1 = f1_score(y_train, y_pred_cv, average='weighted', zero_division=0)

    print(f"CV Accuracy : {acc:.4f}")
    print(f"CV Precision: {prec:.4f}")
    print(f"CV Recall   : {rec:.4f}")
    print(f"CV F1-score : {f1:.4f}")
    print("\nCV Classification Report:\n", classification_report(y_train, y_pred_cv, zero_division=0))

    cm = confusion_matrix(y_train, y_pred_cv)
    plt.figure(figsize=(10, 7)) # Sınıf sayısı fazlaysa boyutu artırın
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=model.classes_ if hasattr(model, 'classes_') else 'auto', yticklabels=model.classes_ if hasattr(model, 'classes_') else 'auto')
    plt.title(f'CV Confusion Matrix - {name}')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    plt.show()

    cv_results.append({
        "Model": name,
        "CV Accuracy": acc,
        "CV Precision": prec,
        "CV Recall": rec,
        "CV F1-Score": f1
    })

cv_results_df = pd.DataFrame(cv_results)
print("\n--- Çapraz Doğrulama Sonuçları Tablosu ---")
print(cv_results_df)

# Öğrenme Eğrisi Fonksiyonu
def plot_learning_curve_custom(estimator, title, X, y, cv, scoring='accuracy', n_jobs=-1, train_sizes=np.linspace(0.1, 1.0, 5)):
    plt.figure(figsize=(8, 5))
    plt.title(title)
    plt.xlabel("Eğitim Örnek Sayısı")
    plt.ylabel(scoring.capitalize())

    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, scoring=scoring, n_jobs=n_jobs, train_sizes=train_sizes, shuffle=True, random_state=42
    )

    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std  = np.std(train_scores, axis=1)
    test_scores_mean  = np.mean(test_scores, axis=1)
    test_scores_std   = np.std(test_scores, axis=1)

    plt.grid(True)
    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1, color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r", label="Eğitim skoru")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g", label="Çapraz Doğrulama skoru")
    plt.legend(loc="best")
    plt.tight_layout()
    plt.show()

print("\n--- Öğrenme Eğrileri (Eğitim Verisi Üzerinde) ---")
# Öğrenme eğrileri eğitim verisi ve kfold kullanılarak çizilir
for name, model_instance in models.items():
    plot_learning_curve_custom(model_instance,
                               f"Learning Curve - {name}",
                               X_train_scaled, y_train, cv=kfold, scoring='f1_weighted')


# 11. 🧪 Test Seti Üzerinde Model Performansını Değerlendirme Fonksiyonu
def evaluate_model_on_test_set(model_name, model_instance, X_train, y_train, X_test, y_test):
    """
    Verilen modeli eğitim seti üzerinde eğitir ve test seti üzerinde değerlendirir.
    """
    print(f"\n--- Test Ediliyor: {model_name} ---")

    # Modeli tüm eğitim verisi üzerinde eğit
    model_instance.fit(X_train, y_train)

    # Test seti üzerinde tahmin yap
    y_pred_test = model_instance.predict(X_test)

    # Test seti metrikleri
    test_acc = accuracy_score(y_test, y_pred_test)
    test_prec = precision_score(y_test, y_pred_test, average='weighted', zero_division=0)
    test_rec = recall_score(y_test, y_pred_test, average='weighted', zero_division=0)
    test_f1 = f1_score(y_test, y_pred_test, average='weighted', zero_division=0)

    print(f"Test Accuracy : {test_acc:.4f}")
    print(f"Test Precision: {test_prec:.4f}")
    print(f"Test Recall   : {test_rec:.4f}")
    print(f"Test F1-score : {test_f1:.4f}")
    print("\nTest Seti Sınıflandırma Raporu:\n", classification_report(y_test, y_pred_test, zero_division=0, target_names=np.unique(y_test).astype(str))) # unique etiketleri ekle

    # Test seti için confusion matrix
    cm_test = confusion_matrix(y_test, y_pred_test, labels=model_instance.classes_ if hasattr(model_instance, 'classes_') else np.unique(y_test))
    plt.figure(figsize=(10, 7)) # Sınıf sayısı fazlaysa boyutu artırın
    sns.heatmap(cm_test, annot=True, fmt='d', cmap='Oranges', xticklabels=model_instance.classes_ if hasattr(model_instance, 'classes_') else np.unique(y_test).astype(str), yticklabels=model_instance.classes_ if hasattr(model_instance, 'classes_') else np.unique(y_test).astype(str))
    plt.title(f'Test Seti Confusion Matrix - {model_name}')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    plt.show()

    return {
        "Model": model_name,
        "Test Accuracy": test_acc,
        "Test Precision": test_prec,
        "Test Recall": test_rec,
        "Test F1-Score": test_f1
    }

# 12. 📊 Test Seti Sonuçları
test_results = []
print("\n--- Test Seti Değerlendirme Sonuçları ---")
for name, model in models.items():
    # Her model için yeni bir örnek oluşturmak daha güvenli olabilir,
    # ancak cross_val_predict modelin kendisini kalıcı olarak değiştirmez.
    # Eğer emin olmak isterseniz, model tanımlarını burada yeniden yapabilirsiniz.
    # örn: test_model = KNeighborsClassifier(n_neighbors=5)
    test_model_instance = model # Mevcut model örneğini kullanıyoruz
    
    # SVC için classes_ özniteliğini ayarlamak gerekebilir eğer y_test'te olmayan sınıflar varsa
    # Bu genellikle bir sorun olmamalı eğer stratify doğru çalıştıysa ve veri yeterliyse
    # Ancak çok nadir durumlarda veya çok küçük veri setlerinde sorun olabilir.
    # SVC gibi bazı modeller için .fit() sonrası .classes_ oluşur.
    
    result = evaluate_model_on_test_set(name, test_model_instance, X_train_scaled, y_train, X_test_scaled, y_test)
    test_results.append(result)

test_results_df = pd.DataFrame(test_results)
print("\n--- Test Seti Sonuçları Tablosu ---")
print(test_results_df)

# 13. 📊 Test Seti Doğruluk Grafiği
plt.figure(figsize=(10, 6))
sns.barplot(x='Model', y='Test Accuracy', data=test_results_df, palette='viridis')
plt.title('Modellerin Test Seti Doğruluk Karşılaştırması')
plt.xlabel('Model')
plt.ylabel('Test Accuracy')
plt.xticks(rotation=45, ha='right')
plt.ylim(0, 1.05) # Y eksenini 0 ile 1 arasında ayarla
plt.tight_layout()
plt.show()

# 14. ⭐ En İyi Model (Test Seti Doğruluğuna Göre)
if not test_results_df.empty:
    best_model_test = test_results_df.loc[test_results_df['Test Accuracy'].idxmax()]
    print("\n--- Test Seti Üzerindeki En İyi Model ---")
    print(best_model_test)
else:
    print("\nTest sonuçları boş, en iyi model belirlenemedi.")

print("\nAnaliz tamamlandı.")