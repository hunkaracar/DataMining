import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# Veri setini oku
df = pd.read_csv('Global_Cybersecurity_Threats_2015-2024.csv')

print("📊 Veri seti boyutu:", df.shape)
print(df.head(10))

pd.set_option('display.max_rows', 100)
pd.set_option('display.float_format', '{:.2f}'.format)


#Missing values
print("Missing Values:\n", df.isnull().sum())

#info
print(df.info())

# describe data
print("\n📊 Veri seti istatistikleri:")
print(df.describe())

print("\n")
print(df['Year'].value_counts())
print("\n")
print(df['Country'].value_counts())
print("\n")
print(df['Attack Type'].value_counts())
print("\n")
print(df['Target Industry'].value_counts())
print("\n")
print(df['Attack Source'].value_counts())
print("\n")
print(df['Security Vulnerability Type'].value_counts())
print("\n")
print(df['Defense Mechanism Used'].value_counts())
print("\n")

attack_type_counts = df['Attack Type'].value_counts()
# Plot: Number of Attacks by Type
plt.figure(figsize=(10, 6))
sns.barplot(x=attack_type_counts.values, y=attack_type_counts.index)
plt.title('Number of Cybersecurity Attacks by Type (2015-2024)')
plt.xlabel('Number of Attacks')
plt.ylabel('Attack Type')
plt.tight_layout()
plt.show()


# Financial loss by attack type
loss_by_attack = df.groupby('Attack Type')['Financial Loss (in Million $)'].mean().sort_values(ascending=False)
# Plot: Average Financial Loss by Attack Type
plt.figure(figsize=(10, 6))
sns.barplot(x=loss_by_attack.values, y=loss_by_attack.index)
plt.title('Average Financial Loss by Attack Type (2015-2024)')
plt.xlabel('Average Financial Loss (Million $)')
plt.ylabel('Attack Type')
plt.tight_layout()
plt.show()


# Count of attacks by industry
industry_counts = df['Target Industry'].value_counts()
# Plot: Attacks by Industry
plt.figure(figsize=(12, 6))
sns.barplot(x=industry_counts.values, y=industry_counts.index)
plt.title('Number of Attacks by Target Industry (2015-2024)')
plt.xlabel('Number of Attacks')
plt.ylabel('Industry')
plt.tight_layout()
plt.show()

# Financial loss by industry
loss_by_industry = df.groupby('Target Industry')['Financial Loss (in Million $)'].mean().sort_values(ascending=False)
# Plot: Average Financial Loss by Industry
plt.figure(figsize=(12, 6))
sns.barplot(x=loss_by_industry.values, y=loss_by_industry.index)
plt.title('Average Financial Loss by Industry (2015-2024)')
plt.xlabel('Average Financial Loss (Million $)')
plt.ylabel('Industry')
plt.tight_layout()
plt.show()

# Top 10 countries by number of attacks
top_countries = df['Country'].value_counts().head(10)
# Plot: Top 10 Countries by Number of Attacks
plt.figure(figsize=(10, 6))
sns.barplot(x=top_countries.values, y=top_countries.index)
plt.title('Top 10 Countries by Number of Cybersecurity Attacks (2015-2024)')
plt.xlabel('Number of Attacks')
plt.ylabel('Country')
plt.tight_layout()
plt.show()


# Number of attacks per year
attacks_by_year = df['Year'].value_counts().sort_index()
# Plot: Number of Attacks by Year
plt.figure(figsize=(10, 6))
sns.lineplot(x=attacks_by_year.index, y=attacks_by_year.values, marker='o')
plt.title('Number of Cybersecurity Attacks by Year (2015-2024)')
plt.xlabel('Year')
plt.ylabel('Number of Attacks')
plt.tight_layout()
plt.show()


# Average financial loss by year
loss_by_year = df.groupby('Year')['Financial Loss (in Million $)'].mean()
# Plot: Average Financial Loss by Year
plt.figure(figsize=(10, 6))
sns.lineplot(x=loss_by_year.index, y=loss_by_year.values, marker='o')
plt.title('Average Financial Loss by Year (2015-2024)')
plt.xlabel('Year')
plt.ylabel('Average Financial Loss (Million $)')
plt.tight_layout()
plt.show()

# Count of attacks by vulnerability type
vulnerability_counts = df['Security Vulnerability Type'].value_counts()
# Plot: Attacks by Vulnerability Type
plt.figure(figsize=(10, 6))
sns.barplot(x=vulnerability_counts.values, y=vulnerability_counts.index)
plt.title('Number of Attacks by Security Vulnerability Type (2015-2024)')
plt.xlabel('Number of Attacks')
plt.ylabel('Vulnerability Type')
plt.tight_layout()
plt.show()

# Count of attacks by source
source_counts = df['Attack Source'].value_counts()
# Plot: Attacks by Source
plt.figure(figsize=(10, 6))
sns.barplot(x=source_counts.values, y=source_counts.index)
plt.title('Number of Attacks by Attack Source (2015-2024)')
plt.xlabel('Number of Attacks')
plt.ylabel('Attack Source')
plt.tight_layout()
plt.show()

# Correlation between numerical columns
correlation_matrix = df[['Financial Loss (in Million $)', 'Number of Affected Users', 'Incident Resolution Time (in Hours)']].corr()
# Plot: Correlation Heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
plt.title('Correlation Matrix of Numerical Variables')
plt.tight_layout()
plt.show()

# Box Plot: Financial Loss by Target Industry ---
plt.figure(figsize=(14, 8))
sns.boxplot(data=df, x='Target Industry', y='Financial Loss (in Million $)')
plt.title('Distribution of Financial Loss by Target Industry (2015-2024)')
plt.xlabel('Target Industry')
plt.ylabel('Financial Loss (Million $)')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Line Plot: Average Financial Loss by Attack Source Over Years ---
loss_by_source_year = df.groupby(['Year', 'Attack Source'])['Financial Loss (in Million $)'].mean().unstack()
# Plot: Line Plot
plt.figure(figsize=(12, 8))
loss_by_source_year.plot(kind='line', marker='o')
plt.title('Average Financial Loss by Attack Source Over Years (2015-2024)')
plt.xlabel('Year')
plt.ylabel('Average Financial Loss (Million $)')
plt.legend(title='Attack Source', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()

# Average resolution time by defense mechanism
resolution_by_defense = df.groupby('Defense Mechanism Used')['Incident Resolution Time (in Hours)'].mean().sort_values()
# Plot: Average Resolution Time by Defense Mechanism
plt.figure(figsize=(10, 6))
sns.barplot(x=resolution_by_defense.values, y=resolution_by_defense.index)
plt.title('Average Incident Resolution Time by Defense Mechanism (2015-2024)')
plt.xlabel('Average Resolution Time (Hours)')
plt.ylabel('Defense Mechanism')
plt.tight_layout()
plt.show()

# Nation-State kaynaklı saldırılar
nation_state_attacks = df[df['Attack Source'] == 'Nation-state']

# Hedef alınan endüstriler
industry_targeted = nation_state_attacks['Target Industry'].value_counts()
print("🌐 Nation-State kaynaklı saldırılarda hedef alınan endüstriler:")
print(industry_targeted)

# Grafik
plt.figure(figsize=(10, 6))
sns.barplot(x=industry_targeted.values, y=industry_targeted.index)
plt.title('Nation-State Saldırılarında Hedef Alınan Endüstriler (2015-2024)')
plt.xlabel('Saldırı Sayısı')
plt.ylabel('Hedef Endüstri')
plt.tight_layout()
plt.show()

attack_type_ns = nation_state_attacks['Attack Type'].value_counts()
print("\n🛡️ Nation-State kaynaklı saldırılarda kullanılan saldırı tipleri:")
print(attack_type_ns)

# Grafik
plt.figure(figsize=(10, 6))
sns.barplot(x=attack_type_ns.values, y=attack_type_ns.index)
plt.title('Nation-State Saldırılarında Kullanılan Saldırı Tipleri (2015-2024)')
plt.xlabel('Saldırı Sayısı')
plt.ylabel('Saldırı Tipi')
plt.tight_layout()
plt.show()


# Finansal kayıp analizi
loss_ns = nation_state_attacks.groupby('Attack Type')['Financial Loss (in Million $)'].mean().sort_values(ascending=False)
print("\n💸 Nation-State kaynaklı saldırılarda ortalama finansal kayıplar (saldırı tipine göre):")
print(loss_ns)

# Grafik
plt.figure(figsize=(10, 6))
sns.barplot(x=loss_ns.values, y=loss_ns.index)
plt.title('Nation-State Saldırılarında Ortalama Finansal Kayıplar (2015-2024)')
plt.xlabel('Ortalama Finansal Kayıp (Milyon $)')
plt.ylabel('Saldırı Tipi')
plt.tight_layout()
plt.show()

# Attack Source'a göre saldırı sayısı ve ortalama finansal kayıp
attack_source_stats = df.groupby('Attack Source')['Financial Loss (in Million $)'].agg(['count', 'mean']).sort_values(by='count', ascending=False)

# Terminale yazdır
print("\n🔍 Attack Source'lara göre istatistikler (Saldırı Sayısı ve Ortalama Finansal Kayıp):\n")
print(attack_source_stats)

# Görselleştirme: Ortalama Finansal Kayıp (Attack Source'a göre)
plt.figure(figsize=(12, 6))
sns.barplot(x=attack_source_stats['mean'], y=attack_source_stats.index)
plt.title('Average Financial Loss by Attack Source (2015-2024)')
plt.xlabel('Average Financial Loss (Million $)')
plt.ylabel('Attack Source')
plt.tight_layout()
plt.show()

# Görselleştirme: Saldırı Sayısı (Attack Source'a göre)
plt.figure(figsize=(12, 6))
sns.barplot(x=attack_source_stats['count'], y=attack_source_stats.index)
plt.title('Number of Attacks by Attack Source (2015-2024)')
plt.xlabel('Number of Attacks')
plt.ylabel('Attack Source')
plt.tight_layout()
plt.show()


# --- 10. Key Insights ---
print("\nAnahtar sonuclar:")
print(f"1. En çok atak tipi: {attack_type_counts.idxmax()} ({attack_type_counts.max()} attacks)")
print(f"2. En yüksek ortalama mali kayba sahip sektör: {loss_by_industry.idxmax()} (${loss_by_industry.max():.2f}M)")
print(f"3. En çok atak alan ülke: {top_countries.idxmax()} ({top_countries.max()} attacks)")
print(f"4. En çok sömürülen güvenlik acigi: {vulnerability_counts.idxmax()} ({vulnerability_counts.max()} attacks)")
print(f"5. En yaygin atak kaynagi: {source_counts.idxmax()} ({source_counts.max()} attacks)")
print(f"6. En hizli savunma mekanizmasi(ortalama cözülme zamani): {resolution_by_defense.idxmin()} ({resolution_by_defense.min():.2f} hours)")

loss_by_attack = df.groupby('Attack Type')['Financial Loss (in Million $)'].mean().sort_values(ascending=False)
print("\n💸 Saldırı Tipine Göre Ortalama Finansal Kayıp:")
print(loss_by_attack)

industry_counts = df['Target Industry'].value_counts()
print("\n🏭 Hedef Endüstrilere Göre Saldırı Sayıları:")
print(industry_counts)

loss_by_industry = df.groupby('Target Industry')['Financial Loss (in Million $)'].mean().sort_values(ascending=False)
print("\n💼 Hedef Endüstrilere Göre Ortalama Finansal Kayıp:")
print(loss_by_industry)

top_countries = df['Country'].value_counts().head(10)
print("\n🌍 En Çok Saldırı Alan İlk 10 Ülke:")
print(top_countries)

attacks_by_year = df['Year'].value_counts().sort_index()
print("\n📅 Yıllara Göre Saldırı Sayısı:")
print(attacks_by_year)

loss_by_year = df.groupby('Year')['Financial Loss (in Million $)'].mean()
print("\n📉 Yıllara Göre Ortalama Finansal Kayıp:")
print(loss_by_year)

vulnerability_counts = df['Security Vulnerability Type'].value_counts()
print("\n🛠️ Güvenlik Açığı Türüne Göre Saldırı Sayıları:")
print(vulnerability_counts)

source_counts = df['Attack Source'].value_counts()
print("\n🧨 Saldırı Kaynağına Göre Saldırı Sayıları:")
print(source_counts)

correlation_matrix = df[['Financial Loss (in Million $)', 'Number of Affected Users', 'Incident Resolution Time (in Hours)']].corr()
print("\n📈 Sayısal Değişkenler Arasındaki Korelasyon:")
print(correlation_matrix)

resolution_by_defense = df.groupby('Defense Mechanism Used')['Incident Resolution Time (in Hours)'].mean().sort_values()
print("\n🛡️ Savunma Mekanizmasına Göre Ortalama Çözülme Süresi:")
print(resolution_by_defense)

industry_targeted = nation_state_attacks['Target Industry'].value_counts()
print("\n🌐 Nation-State kaynaklı saldırılarda hedef alınan endüstriler:")
print(industry_targeted)

attack_type_ns = nation_state_attacks['Attack Type'].value_counts()
print("\n🛡️ Nation-State kaynaklı saldırılarda kullanılan saldırı tipleri:")
print(attack_type_ns)

loss_ns = nation_state_attacks.groupby('Attack Type')['Financial Loss (in Million $)'].mean().sort_values(ascending=False)
print("\n💸 Nation-State kaynaklı saldırılarda ortalama finansal kayıplar (saldırı tipine göre):")
print(loss_ns)

attack_source_stats = df.groupby('Attack Source')['Financial Loss (in Million $)'].agg(['count', 'mean']).sort_values(by='count', ascending=False)
print("\n🔍 Attack Source'lara göre istatistikler (Saldırı Sayısı ve Ortalama Finansal Kayıp):")
print(attack_source_stats)

