# 💳 Sentinel Guard: AI Fraud Detection System

Bu proje, Kaggle Credit Card Fraud Detection veri setini kullanarak kredi kartı işlemlerindeki dolandırıcılıkları (fraud) tespit eden uçtan uca bir makine öğrenmesi çözümüdür. Sistem, hem model eğitim sürecini hem de son kullanıcı için interaktif bir analiz arayüzünü (Dashboard) içerir.

## 🚀 Öne Çıkan Özellikler

- **Dengesiz Veri Yönetimi:** SMOTE (Synthetic Minority Over-sampling Technique) ile düşük orandaki fraud verileri dengelenmiştir.
- **Güçlü Algoritmalar:** Logistic Regression ve Regularized Random Forest modelleri karşılaştırmalı olarak eğitilmiştir.
- **Overfitting Koruması:** Yinelenen kayıt (duplicate) temizliği, Stratified K-Fold Cross-Validation ve derinlik kısıtlamalı ağaçlar kullanılmıştır.
- **Gelişmiş Metrikler:** Sadece Accuracy değil; Precision, Recall, F1-Score ve Precision-Recall (PR) Curve analizi ile gerçek başarı ölçülmüştür.
- **İnteraktif Dashboard:** Streamlit tabanlı arayüz ile CSV dosyası yükleyip anında şüpheli işlem tespiti yapılabilir.

## 🛠️ Kurulum

Proje Python 3.12+ ile geliştirilmiştir. Gerekli kütüphaneleri yüklemek için:

```powershell
pip install -r requirements.txt
```

_(Not: Eğer requirements.txt yoksa şu komutu kullanın: `pip install pandas numpy matplotlib seaborn scikit-learn imbalanced-learn joblib streamlit tabulate`)_

## 📂 Kullanım

### 1. Modelin Eğitilmesi

Veri setini analiz etmek ve en iyi modeli oluşturmak için:

```powershell
python fraud_detection.py
```

Bu komut sonunda `best_model.pkl` dosyası oluşturulacak, performans grafikleri (ROC, PR Curve) kaydedilecektir.

### 2. Dashboard'un Başlatılması

Eğitilmiş modeli kullanarak şüpheli işlemleri arayüz üzerinden görmek için:

```powershell
streamlit run app.py
```

## 📉 Dosya Yapısı

- `fraud_detection.py`: Veri işleme, model eğitimi ve overfitting kontrollerini içeren ana script.
- `app.py`: Streamlit tabanlı kullanıcı arayüzü.
- `dataset/`: `creditcard.csv` dosyasının bulunduğu dizin.
- `best_model.pkl`: Eğitilmiş ve kaydedilmiş en iyi performanslı model.
- `model_performance_comparison.png`: Modellerin başarı grafiklerini gösteren görsel.

## ⚖️ Lisans

Bu proje eğitim ve portfolyo amaçlı geliştirilmiştir. Veri seti Kaggle üzerinden temin edilmiştir.

---

**Sentinel Guard** - _Yapay Zeka ile Güvenli İşlemler._
