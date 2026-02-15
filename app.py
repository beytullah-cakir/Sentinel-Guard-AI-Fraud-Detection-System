import streamlit as st
import pandas as pd
import joblib
import os

# Sayfa ayarları
st.set_page_config(page_title="Fraud Detection Dashboard", page_icon="💳", layout="wide")

# CSS ile Premium Görünüm
st.markdown("""
    <style>
    .main {
        background-color: #f5f7f9;
    }
    .stButton>button {
        width: 100%;
        border-radius: 5px;
        height: 3em;
        background-color: #ff4b4b;
        color: white;
    }
    .fraud-card {
        padding: 20px;
        border-radius: 10px;
        background-color: #ffffff;
        border-left: 5px solid #ff4b4b;
        box-shadow: 2px 2px 10px rgba(0,0,0,0.1);
        margin-bottom: 10px;
    }
    .safe-card {
        padding: 20px;
        border-radius: 10px;
        background-color: #ffffff;
        border-left: 5px solid #2ecc71;
        box-shadow: 2px 2px 10px rgba(0,0,0,0.1);
        margin-bottom: 10px;
    }
    </style>
    """, unsafe_allow_html=True)

def load_model():
    if os.path.exists('best_model.pkl'):
        return joblib.load('best_model.pkl')
    return None

def main():
    st.title("💳 Kredi Kartı Dolandırıcılık Tespit Sistemi")
    st.markdown("Eğitilmiş yapay zeka modelini kullanarak şüpheli işlemleri anında tespit edin.")

    model = load_model()
    
    if model is None:
        st.error("❌ Model dosyası (best_model.pkl) bulunamadı! Lütfen önce training kodunu çalıştırın.")
        return

    # Sidebar - Dosya Yükleme
    st.sidebar.header("📁 Veri Yükleme")
    uploaded_file = st.sidebar.file_uploader("CSV dosyasını seçin", type=["csv"])

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        
        st.subheader("📊 Yüklenen Veri Önizlemesi")
        st.dataframe(df.head(10))

        # Tahmin butonu
        if st.button("Şüpheli İşlemleri Analiz Et"):
            # Model eğitimi sırasında kullanılan feature listesi (V1-V28 + Amount)
            # Time sütunu varsa çıkarıyoruz
            process_df = df.copy()
            if 'Time' in process_df.columns:
                process_df = process_df.drop(['Time'], axis=1)
            
            # Eğer Class sütunu varsa onu da tahmin için çıkarıyoruz
            if 'Class' in process_df.columns:
                process_df = process_df.drop(['Class'], axis=1)

            # Tahminleri yap
            predictions = model.predict(process_df)
            probs = model.predict_proba(process_df)[:, 1]

            df['Fraud_Probability'] = probs
            df['Prediction'] = predictions

            # Sonuçları filtrele (Fraud tahmin edilenler)
            fraudulent_trans = df[df['Prediction'] == 1].sort_values(by='Fraud_Probability', ascending=False)

            st.divider()
            
            # Metrikler
            col1, col2, col3 = st.columns(3)
            col1.metric("Toplam İşlem", len(df))
            col2.metric("Tespit Edilen Şüpheli İşlem", len(fraudulent_trans))
            col3.metric("Normal İşlem", len(df) - len(fraudulent_trans))

            if len(fraudulent_trans) > 0:
                st.warning(f"⚠️ Toplam {len(fraudulent_trans)} adet şüpheli işlem tespit edildi!")
                
                st.subheader("🕵️ Şüpheli Kayıtlar Listesi")
                # Kullanıcıya daha anlamlı göstermek için olasılıkla birlikte gösterelim
                display_cols = ['Amount'] + [c for c in df.columns if c.startswith('V')] + ['Fraud_Probability']
                st.dataframe(fraudulent_trans[display_cols].style.background_gradient(subset=['Fraud_Probability'], cmap='Reds'))
                
                # İndirme seçeneği
                csv = fraudulent_trans.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="Şüpheli İşlemleri CSV Olarak İndir",
                    data=csv,
                    file_name='fraudulent_transactions_report.csv',
                    mime='text/csv',
                )
            else:
                st.success("✅ Harika! Hiçbir şüpheli işlem tespit edilmedi.")

    else:
        st.info("💡 Analiz yapmak için sol taraftaki panelden bir CSV dosyası yükleyin.")
        st.markdown("""
        ### Dosya Formatı Hakkında:
        Yükleyeceğiniz CSV dosyası eğitilen modelin beklediği şu sütunları içermelidir:
        - `V1`'den `V28`'e kadar anonim özellikler.
        - `Amount`: İşlem tutarı.
        - *(Opsiyonel)* `Time`: İşlem zamanı.
        """)

if __name__ == "__main__":
    main()
