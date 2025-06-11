import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_predict, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier  # 💡 Eklendi
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import learning_curve

# 1. 📁 Veri setinin yolu
file_path = 'Global_Cybersecurity_Threats_2015-2024.csv'

# 2. 📊 Veri setini oku
df = pd.read_csv(file_path)

# 3. 🎯 Hedef değişken
y = df['Target Industry']  # Hangi sektöre saldırıldığı tahmin edilecek
# Target Industry → KNN 0.968
# Attack Type --> KNN 0.8456
# Security Vulnerability Type --> KNN 0.862333
print(f"Class count: {y.nunique()}")


# 4. 🔢 Kategorik verileri sayısala çevir (One-Hot Encoding)
df_encoded = pd.get_dummies(df.drop(['Financial Loss (in Million $)'], axis=1))

# 5. ✅ Özellik matrisi (X)
X = df_encoded

# 6. 🔧 Veriyi ölçeklendir (standartlaştır)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 7. 🔁 Stratified K-Fold çapraz doğrulama
kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
# StratifiedKFold → veriyi 5 parçaya böler, her parçada sınıf oranlarını (label distribution) korur.

# 8. 🤖 Kullanılacak modeller
models = {
    "KNN (k=5)": KNeighborsClassifier(n_neighbors=5),
    "SVC (linear)": SVC(kernel='linear', probability=True),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42), 
}


# 9. 📈 Sonuçları kaydetmek için liste
results = []

# 10. 🔁 Her model için değerlendirme
for name, model in models.items():
    print(f"\nModel: {name}")
    
    # 🔄 K-Fold doğrulama tahminleri
    y_pred = cross_val_predict(model, X_scaled, y, cv=kfold)
    # model: eğitim + tahmin burada yapılır, 5 farklı parçanın her birinde
    # Her parça sırayla test seti olur, diğerleri eğitim seti
    # cross_val_predict → tüm veri için tahminleri döndürür (çapraz doğrulama içinde)

    # 🎯 Değerlendirme metrikleri
    acc = accuracy_score(y, y_pred)
    prec = precision_score(y, y_pred, average='weighted', zero_division=0)
    rec = recall_score(y, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y, y_pred, average='weighted', zero_division=0)

    # 🖨️ Metrik çıktıları
    print("Accuracy :", f"{acc:.4f}")
    print("Precision:", f"{prec:.4f}")
    print("Recall   :", f"{rec:.4f}")
    print("F1-score :", f"{f1:.4f}")

    # 📝 Sınıflandırma raporu
    print("\nClassification Report:\n", classification_report(y, y_pred, zero_division=0))

    # 📊 Confusion matrix
    cm = confusion_matrix(y, y_pred)
    plt.figure(figsize=(8, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Confusion Matrix - {name}')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    plt.show()

    # 📋 Sonuçları tabloya ekle
    results.append({
        "Model": name,
        "Accuracy": acc,
        "Precision": prec,
        "Recall": rec,
        "F1-Score": f1
    })


def plot_learning_curve(estimator, title, X, y, cv, scoring='accuracy', n_jobs=-1, train_sizes=np.linspace(0.1, 1.0, 5)):
    """
    Verilen model için öğrenme eğrisi çizer.
    """
    plt.figure(figsize=(8, 5))
    plt.title(title)
    plt.xlabel("Training Sample Size")
    plt.ylabel(scoring.capitalize())
    
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, scoring=scoring, n_jobs=n_jobs, train_sizes=train_sizes
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

    plt.plot(train_sizes, train_scores_mean, 'o-', color="r", label="Train Score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g", label="Validation Score")

    plt.legend(loc="best")
    plt.tight_layout()
    plt.show()

# Öğrenme eğrisi - Random Forest
plot_learning_curve(RandomForestClassifier(n_estimators=100, random_state=42),
                    "Learning Curve - Random Forest", X_scaled, y, cv=kfold, scoring='f1_weighted')

# Öğrenme eğrisi - SVC
plot_learning_curve(SVC(kernel='linear', probability=True),
                    "Learning Curve - SVC (linear)", X_scaled, y, cv=kfold, scoring='f1_weighted')

# Öğrenme eğrisi - SVC
plot_learning_curve(KNeighborsClassifier(n_neighbors=5),
                    "Learning Curve - KNN", X_scaled, y, cv=kfold, scoring='f1_weighted')


# üç model için accuracy grafiği
results_df = pd.DataFrame(results)
plt.figure(figsize=(10, 6))
sns.barplot(x='Model', y='Accuracy', data=results_df, palette='viridis')
plt.title('Model Accuracy Comparison')
plt.xlabel('Model')
plt.ylabel('Accuracy')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
# Sonuçları tabloya yazdır
print("\nSonuçlar:")
print(results_df)
# En iyi model
best_model = results_df.loc[results_df['Accuracy'].idxmax()]
print("\nEn iyi model:")
print(best_model)
# En iyi modelin detayları

# ✅ En iyi modeli tekrar tanımla ve eğit
#best_clf = KNeighborsClassifier(n_neighbors=5)
best_clf = SVC(kernel='linear')
best_clf.fit(X_scaled, y)

# 🔎 Kullanıcıdan test verisi almak için örnek veri tanımı
# Bu veri, orijinal veri setindeki sütunlarla uyumlu olmalıdır.
# Örnek: bir siber saldırının detaylarını simüle ediyoruz.
# Not: Gerçek kullanımda bu veriyi formdan veya sistemden alabilirsin.

"""
test_data_dict = {
    'Year': 2024,
    'Country': 'USA',
    'Attack Type': 'Phishing',
    'Security Vulnerability Type': 'Email Spoofing',
    'Severity': 'High',
    'Attack Vector': 'Email',
    'Threat Actor Type': 'Criminal Group',
    'Record Count Leaked': 120000,
    'Detection Method': 'Anomaly Detection',
    'Target Industry': 'Unknown'  # Bu sadece y'de olduğu için çıkarılabilir
}

"""
test_data_dict = {
    'Year': 2021,
    'Country': 'India',
    'Attack Type': 'SQL Injection',
    'Security Vulnerability Type': 'Input Validation Failure',
    'Severity': 'Medium',
    'Attack Vector': 'Web',
    'Threat Actor Type': 'Hacktivist',
    'Record Count Leaked': 10000,
    'Detection Method': 'Manual Review',
    'Target Industry': 'Unknown'
}

# 🔧 Test verisini DataFrame'e çevir
test_df = pd.DataFrame([test_data_dict])

# 🔢 One-Hot Encoding işlemi (eğitimdeki sütunlarla uyum sağlamak için)
test_encoded = pd.get_dummies(test_df.drop(['Target Industry'], axis=1))

# 🧩 Eksik sütunları tamamlama (eğitimdeki ile uyumsuzluk varsa)
for col in df_encoded.columns:
    if col not in test_encoded.columns:
        test_encoded[col] = 0

# ✅ Sütun sırasını aynı hale getir
test_encoded = test_encoded[df_encoded.columns]

# 🔧 Test verisini ölçeklendir
test_scaled = scaler.transform(test_encoded)

# 🔮 Tahmin et
predicted_sector = best_clf.predict(test_scaled)[0]

# 🖨️ Sonucu göster
print("\n🔍 Tahmin edilen hedef sektör:")
print(predicted_sector)
