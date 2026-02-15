import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
import joblib
import os

# --- 1. AYARLAR VE GÖRSELLEŞTİRME AYARLARI ---
# Grafiklerin stilini ayarlayalım
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

# Dataset yolu
csv_path = "dataset/creditcard.csv"

def run_fraud_detection():
    # --- 2. VERİ YÜKLEME ---
    print("--- 2. Veri Yükleme ---")
    if not os.path.exists(csv_path):
        print(f"Hata: {csv_path} bulunamadı!")
        return

    df = pd.read_csv(csv_path)
    print(f"Veri seti yüklendi. Orijinal Satır/Sütun: {df.shape}")

    # Overfitting Önlemi: Yinelenen satırları kontrol et ve temizle
    duplicates = df.duplicated().sum()
    if duplicates > 0:
        print(f"Uyarı: {duplicates} adet yinelenen satır bulundu. Temizleniyor...")
        df = df.drop_duplicates()
        print(f"Yinelenenler sonrası satır sayısı: {df.shape[0]}")

    # Fraud oranını ekrana yazdır (Class = 1)
    fraud_count = df[df['Class'] == 1].shape[0]
    total_count = df.shape[0]
    fraud_ratio = (fraud_count / total_count) * 100
    print(f"Toplam İşlem: {total_count}")
    print(f"Fraud İşlem Sayısı: {fraud_count}")
    print(f"Fraud Oranı: %{fraud_ratio:.4f}\n")

    # --- 3. FEATURE SEÇİMİ ---
    print("--- 3. Feature Seçimi ---")
    X = df.drop(['Time', 'Class'], axis=1)
    y = df['Class']

    # --- 4. TRAIN/TEST SPLIT ---
    print("--- 4. Train/Test Split ---")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    # --- 5. STANDARDİZASYON ---
    print("--- 5. Standardizasyon ---")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # --- 6. MODEL 1: LOGISTIC REGRESSION ---
    print("--- 6. Model 1: Logistic Regression ---")
    lr_model = LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42)
    lr_model.fit(X_train_scaled, y_train)
    
    y_pred_lr = lr_model.predict(X_test_scaled)
    y_proba_lr = lr_model.predict_proba(X_test_scaled)[:, 1]

    # --- 7. MODEL 2: SMOTE + RANDOM FOREST (Regularized) ---
    print("\n--- 7. Model 2: SMOTE + Random Forest (Düzenlenmiş) ---")
    sm = SMOTE(random_state=42)
    X_train_res, y_train_res = sm.fit_resample(X_train_scaled, y_train)

    # Overfitting'i önlemek için max_depth ve min_samples_leaf ekledik
    rf_model = RandomForestClassifier(
        n_estimators=100, 
        max_depth=10, 
        min_samples_leaf=5,
        random_state=42, 
        n_jobs=-1
    )
    rf_model.fit(X_train_res, y_train_res)

    y_pred_rf = rf_model.predict(X_test_scaled)
    y_proba_rf = rf_model.predict_proba(X_test_scaled)[:, 1]

    # --- 8. GRAFİKLER (ROC ve Precision-Recall) ---
    print("\n--- 8. Grafiklerin Hazırlanması ---")
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # ROC Curve
    fpr_lr, tpr_lr, _ = metrics.roc_curve(y_test, y_proba_lr)
    fpr_rf, tpr_rf, _ = metrics.roc_curve(y_test, y_proba_rf)
    lr_auc = metrics.roc_auc_score(y_test, y_proba_lr)
    rf_auc = metrics.roc_auc_score(y_test, y_proba_rf)

    ax1.plot(fpr_lr, tpr_lr, label=f'LR (AUC = {lr_auc:.4f})')
    ax1.plot(fpr_rf, tpr_rf, label=f'RF (AUC = {rf_auc:.4f})')
    ax1.plot([0, 1], [0, 1], 'k--')
    ax1.set_title('ROC Curve')
    ax1.legend()

    # Precision-Recall Curve (Imbalanced veriler için daha güvenilir)
    prec_lr, rec_lr, _ = metrics.precision_recall_curve(y_test, y_proba_lr)
    prec_rf, rec_rf, _ = metrics.precision_recall_curve(y_test, y_proba_rf)
    lr_ap = metrics.average_precision_score(y_test, y_proba_lr)
    rf_ap = metrics.average_precision_score(y_test, y_proba_rf)

    ax2.plot(rec_lr, prec_lr, label=f'LR (AP = {lr_ap:.4f})')
    ax2.plot(rec_rf, prec_rf, label=f'RF (AP = {rf_ap:.4f})')
    ax2.set_title('Precision-Recall Curve')
    ax2.set_xlabel('Recall')
    ax2.set_ylabel('Precision')
    ax2.legend()

    plt.tight_layout()
    plt.savefig('model_performance_comparison.png')
    plt.show()

    # --- 9. MODEL KARŞILAŞTIRMA VE CROSS-VALIDATION ---
    print("\n--- 9. Cross-Validation (Overfitting Kontrolü) ---")
    from sklearn.model_selection import cross_val_score
    
    # Sadece küçük bir kısmı üzerinde CV yaparak hız kazanalım (isteğe bağlı)
    # n_jobs=-1 ile hızlandırıyoruz
    cv_scores = cross_val_score(rf_model, X_train_scaled, y_train, cv=3, scoring='roc_auc', n_jobs=-1)
    print(f"Random Forest CV ROC-AUC Skorları: {cv_scores}")
    print(f"Ortalama CV Skor: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")

    p_lr, r_lr, f1_lr, _ = metrics.precision_recall_fscore_support(y_test, y_pred_lr, average='binary')
    p_rf, r_rf, f1_rf, _ = metrics.precision_recall_fscore_support(y_test, y_pred_rf, average='binary')

    results = pd.DataFrame({
        'Model': ['Logistic Regression', 'Random Forest'],
        'Precision': [p_lr, p_rf],
        'Recall': [r_lr, r_rf],
        'F1': [f1_lr, f1_rf],
        'ROC-AUC': [lr_auc, rf_auc],
        'Avg Precision (AP)': [lr_ap, rf_ap]
    })
    print("\nFinal Test Sonuçları:")
    print(results.to_markdown(index=False))

    # --- 10. MODEL KAYDETME ---
    best_model = rf_model if rf_ap > lr_ap else lr_model
    joblib.dump(best_model, 'best_model.pkl')
    print(f"\nEn iyi model kaydedildi. (AP kriterine göre)")

if __name__ == "__main__":
    run_fraud_detection()
