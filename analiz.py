# analiz.py
"""
analiz.py
- Eğitilmiş modeli yükler (bootcamp_project.py tarafından kaydedilen)
- Tek bir görsel veya klasördeki tüm görseller için tahmin yapar
- Tahminleri CSV'ye kaydeder
- Grad-CAM ile görselleştirme yapar ve çıktı resimlerini kaydeder
"""
import os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import load_img, img_to_array

# Ayarlar
MODEL_PATH = Path("models/breakhis_final_model.h5")  # bootcamp_project.py ile kaydedilen model
OUTPUT_DIR = Path("analysis_outputs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
GRADCAM_DIR = OUTPUT_DIR / "gradcam"
GRADCAM_DIR.mkdir(parents=True, exist_ok=True)
PRED_CSV = OUTPUT_DIR / "predictions.csv"
IMG_SIZE = (224, 224)
CLASS_NAMES = ['benign', 'malignant']  # bootcamp_project içinde belirlenen sıraya uy

# Model yükle
if not MODEL_PATH.exists():
    raise SystemExit(f"Model bulunamadı: {MODEL_PATH}. Önce bootcamp_project.py ile eğitip kaydetmelisin.")
print("Model yükleniyor...")
model = load_model(str(MODEL_PATH))
print("Model yüklendi.")

# -------------------------
# Tek görsel tahmin
# -------------------------
def predict_image(img_path):
    img = load_img(img_path, target_size=IMG_SIZE)
    arr = img_to_array(img) / 255.0
    arr = np.expand_dims(arr, axis=0)
    prob = float(model.predict(arr)[0][0])
    label = CLASS_NAMES[int(prob > 0.5)]
    return label, prob

# -------------------------
# Klasör içindeki tüm görselleri tahmin et ve CSV'ye kaydet
# -------------------------
def analyze_folder(folder_path, save_csv=True):
    folder = Path(folder_path)
    rows = []
    for f in sorted(folder.rglob("*")):
        if f.is_file() and f.suffix.lower() in (".png", ".jpg", ".jpeg"):
            try:
                label, prob = predict_image(str(f))
                rows.append({"file": str(f), "pred_label": label, "probability": prob})
                print(f"{f.name} -> {label} ({prob:.4f})")
            except Exception as e:
                print(f"Hata: {f} -> {e}")
    if save_csv:
        df = pd.DataFrame(rows)
        df.to_csv(PRED_CSV, index=False)
        print(f"\nTahminler kaydedildi: {PRED_CSV}")
    return rows

# -------------------------
# Grad-CAM fonksiyonu
# -------------------------
def find_last_conv_layer(model):
    # Modelin son Conv2D katmanını döndür
    for layer in reversed(model.layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            return layer.name
    raise ValueError("Model içinde Conv2D katmanı bulunamadı.")

def make_gradcam_heatmap(img_array, model, last_conv_layer_name):
    """
    img_array: (1, H, W, 3), normalized [0,1]
    döndürür: heatmap (H, W) değerleri 0..1
    """
    grad_model = tf.keras.models.Model(
        [model.inputs], [model.get_layer(last_conv_layer_name).output, model.output]
    )
    with tf.GradientTape() as tape:
        conv_outputs, preds = grad_model(img_array)
        loss = preds[:, 0]  # binary logits (sigmoid output)
    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0].numpy()
    pooled_grads = pooled_grads.numpy()
    for i in range(pooled_grads.shape[-1]):
        conv_outputs[:, :, i] *= pooled_grads[i]
    heatmap = np.mean(conv_outputs, axis=-1)
    heatmap = np.maximum(heatmap, 0)
    maxh = np.max(heatmap) if np.max(heatmap) != 0 else 1e-10
    heatmap /= maxh
    return heatmap

def save_and_show_gradcam(img_path, output_path=None, alpha=0.4):
    """
    Verilen görsel için Grad-CAM üretir, overlay oluşturur ve kaydeder.
    """
    img = cv2.imread(str(img_path))
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, IMG_SIZE)
    img_array = img_to_array(img_resized) / 255.0
    img_batch = np.expand_dims(img_array, axis=0)

    last_conv = find_last_conv_layer(model)
    heatmap = make_gradcam_heatmap(img_batch, model, last_conv)
    heatmap = cv2.resize(heatmap, (img_resized.shape[1], img_resized.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap_colored = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    superimposed = cv2.addWeighted(cv2.cvtColor(img_resized, cv2.COLOR_RGB2BGR), 1-alpha, heatmap_colored, alpha, 0)

    # Kaydet
    if output_path is None:
        output_path = GRADCAM_DIR / f"{Path(img_path).stem}_gradcam.jpg"
    cv2.imwrite(str(output_path), superimposed)
    # Göster
    plt.figure(figsize=(10,5))
    plt.subplot(1,2,1)
    plt.imshow(img_resized)
    plt.title("Orijinal (yeniden boyutlandırılmış)")
    plt.axis('off')
    plt.subplot(1,2,2)
    plt.imshow(cv2.cvtColor(superimposed, cv2.COLOR_BGR2RGB))
    plt.title("Grad-CAM overlay")
    plt.axis('off')
    plt.tight_layout()
    plt.show()

    return output_path

# -------------------------
# Komut satırı arayüzü (basit)
# -------------------------
def interactive_cli():
    print("="*50)
    print("ANALİZ ARACI")
    print("="*50)
    print("1) Tek görsel tahmini ve Grad-CAM")
    print("2) Klasör içi tahmin (tüm görseller -> CSV) ve Grad-CAM üret")
    print("3) Çıkış")
    choice = input("Seçiminiz (1/2/3): ").strip()
    if choice == "1":
        p = input("Görsel yolu: ").strip()
        lab, pr = predict_image(p)
        print(f"Tahmin: {lab} ({pr:.4f})")
        save_and_show_gradcam(p)
    elif choice == "2":
        folder = input("Klasör yolu: ").strip()
        rows = analyze_folder(folder, save_csv=True)
        # Grad-CAM üret (örnek olarak ilk 10 görsele)
        limit = int(input("Kaç görsele Grad-CAM üretmek istersiniz? (ör: 10): ").strip() or "10")
        for i, row in enumerate(rows[:limit]):
            print(f"Grad-CAM {i+1}/{min(limit, len(rows))}: {row['file']}")
            save_and_show_gradcam(row['file'])
    else:
        print("Çıkış yapılıyor.")

if __name__ == "__main__":
    interactive_cli()
