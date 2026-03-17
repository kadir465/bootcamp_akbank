# 🔬 Akbank Derin Öğrenme Bootcamp: Meme Kanseri Histopatolojik Görüntü Sınıflandırması

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)
![Keras Tuner](https://img.shields.io/badge/KerasTuner-Optimized-success.svg)
![Dataset](https://img.shields.io/badge/Dataset-BreaKHis-lightgrey.svg)
![Accuracy](https://img.shields.io/badge/Accuracy-85.3%25-green.svg)

## 📌 Proje Özeti
Bu proje, **Akbank Derin Öğrenme Bootcamp** kapsamında geliştirilmiş, meme kanseri histopatolojik görüntülerini yüksek doğrulukla analiz eden bir sınıflandırma sistemidir. **BreaKHis** veri seti üzerinde **Convolutional Neural Network (CNN)** mimarisi kullanılarak, benign (iyi huylu) ve malignant (kötü huylu) dokuların tam otomatik olarak sınıflandırılması amaçlanmıştır. 

Ayrıca, derin öğrenme modellerinin şeffaflığını artırmak ve tıbbi karar destek süreçlerine entegrasyonunu kolaylaştırmak amacıyla **Grad-CAM (Explainable AI - Açıklanabilir Yapay Zeka)** teknikleri kullanılmıştır.

---

## 🎯 Proje Amacı
Meme kanseri, dünya genelinde kadınlarda en sık görülen kanser türü olup **erken ve doğru teşhis** hayat kurtarıcı bir rol oynamaktadır. Geleneksel histopatolojik incelemeler uzman patologlara bağımlı, zaman alıcı ve göreceli değerlendirmelere açık olabilmektedir. 

Bu projenin temel hedefi:
* Patologların iş yükünü hafifletmek,
* Standartlaşmış, hızlı ve yüksek doğruluklu bir "İkincil Görüş" (Second Opinion) sistemi sunmak,
* Explainable AI (XAI) yaklaşımlarıyla doktorların model kararlarına olan güvenini artırmaktır.

---

## 📊 Veri Seti (BreaKHis)
Çalışmada **[Breast Cancer Histopathological Database (BreaKHis)](https://www.kaggle.com/datasets/ambarish/breakhis)** veri seti kullanılmıştır.

* **Hasta Sayısı:** 82
* **Toplam Görüntü:** 7,909 adet histopatolojik görüntü
* **Çözünürlük ve Büyütme:** 700x460 piksel; 40X, 100X, 200X ve 400X klinik büyütme oranları
* **Sınıf Dağılımı:**
  * 🟢 **Benign (İyi Huylu) - 2,480:** Adenozis, Fibroadenom, Fillodes Tümör, Tübüler Adenom
  * 🔴 **Malignant (Kötü Huylu) - 5,429:** Duktal Karsinom, Lobüler Karsinom, Müsinöz Karsinom, Papiller Karsinom

---

## ⚙️ Yöntem ve Teknolojik Altyapı

* **Diller ve Kütüphaneler:** Python, TensorFlow / Keras, NumPy, Pandas, Matplotlib, Seaborn
* **Veri Ön İşleme (Preprocessing):** Tüm görüntüler model optimizasyonu için `64x64` boyutuna yeniden ölçeklendirilmiş ve normalize edilmiştir (Min-Max Scaling `0-1` aralığında). 
* **Veri Artırma (Data Augmentation):** Aşırı öğrenmeyi (overfitting) önlemek ve genelleme yeteneğini artırmak amacıyla Rotation (döndürme), Shift (kaydırma), Zoom (yakınlaştırma), Flip (çevirme) ve Parlaklık (brightness) ayarı işlemleri uygulanmıştır.

### 🧠 Model Mimarisi
Projede baştan sona (End-to-End) eğitilen özelleştirilmiş bir **Sequential CNN** mimarisi tasarlanmıştır:
1. **Feature Extraction:** İki Convolutional (Evrişim) katmanı
2. **Pooling:** Maksimum Havuzlama (Max Pooling) katmanları
3. **Regularization:** Batch Normalizasyon ve Dropout katmanları
4. **Sınıflandırma (Classification):** Çıkış katmanında ikili sınıflandırma (Binary Classification) işlemi için **Sigmoid** aktivasyon fonksiyonu.

### 🔍 Hiperparametre Optimizasyonu
**Keras Tuner / Random Search** metodu kullanılarak model kapasitesi maksimum seviyeye çıkarılmıştır. Optimize edilen parametreler:
* Evrişim (Conv) ve Yoğun (Dense) katman sayıları,
* Katmanlardaki filtre / nöron sayıları,
* Dropout oranları, Optimizer algoritmaları ve Learning Rate (Öğrenme Oranı).

---

## 📈 Sonuçlar ve Performans

Model doğrulama (validation) seti üzerindeki nihai performansı:

| Metrik | Değer |
| :--- | :--- |
| **Doğruluk (Accuracy)** | `%85.3` |
| **Kesinlik (Precision)** | `%87.1` |
| **Duyarlılık (Recall)** | `%83.5` |

### 💡 Explainable AI (XAI) - Grad-CAM Görselleştirmesi
Modelin "kara kutu" (black-box) yapısını kırmak için uygulanan **Grad-CAM (Gradient-weighted Class Activation Mapping)** tekniği sayesinde:
* Modelin bir hastayı "Malignant" (Tümörlü) olarak tahmin ederken histopatolojik görüntünün hangi bölgelerine odaklandığı, ısı haritaları (Heatmap) aracılığıyla görselleştirilmiştir. 
* Örnek bir görüntü üzerinde model **%92 güven oranıyla** malignant sınıf tahmini yapıp bölgeyi açıkça işaretleyebilmiştir.

---

## 🚀 Kurulum ve Kullanım

Projeyi lokal ortamınızda çalıştırmak için aşağıdaki adımları izleyebilirsiniz:

1. Gerekli kütüphaneleri yükleyin:
   ```bash
   pip install tensorflow keras-tuner numpy pandas matplotlib seaborn opencv-python jupyter
   ```
2. Jupyter Notebook'u başlatın:
   ```bash
   jupyter notebook
   ```
3. `breast_cancer_classification.ipynb` dosyasını açarak hücreleri sırasıyla çalıştırın.

---

## 📎 Bağlantılar & Kaynakça

* 📊 **Veri Seti:** [Kaggle BreaKHis Dataset](https://www.kaggle.com/datasets/ambarish/breakhis)
* 💻 **Kaggle Notebook Yayını:** [Kaggle Projesi - Akademi Versiyonu](https://www.kaggle.com/code/yucelay/akbank-bootcamp1?scriptVersionId=264921293)
