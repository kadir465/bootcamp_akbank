# bootcamp_project.py
"""
Bootcamp proje dosyası
- Veri (train/val/test) oluşturma (rekürsif klasör yapısı destekli)
- tf.data pipeline (augmentasyon, batching, prefetch)
- Model mimarisi (create_model)
- Opsiyonel: Keras Tuner ile hiperparametre arama
- Eğitme, değerlendirme, model kaydetme, eğitim geçmişi kaydetme
"""
import os
from pathlib import Path
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc

# -------------------------------------------------------------------------
# AYARLAR (bunu proje klasörüne göre düzenle)
# -------------------------------------------------------------------------
DATA_DIR = Path(r"C:\programlamapratik\python\bootcamp\BreaKHis_v1")  # veri kök dizini — projene göre güncelle
OUTPUT_DIR = Path("models")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

IMG_SIZE = (224, 224)
BATCH_SIZE = 32
SEED = 42
AUTOTUNE = tf.data.AUTOTUNE
NUM_CLASSES = 2  # binary

# Eğer Keras Tuner kullanmak istiyorsan True yap
USE_KERAS_TUNER = True

# -------------------------------------------------------------------------
# YARDIMCI: klasörden tüm görsel dosyalarını tara ve etiket çıkar
# Bu yapı veri klasöründe 'benign' ve 'malignant' gibi alt klasörler derinlikte olabilir.
# -------------------------------------------------------------------------
IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tiff")

def list_image_paths(data_dir: Path):
    files = []
    for p in data_dir.rglob("*"):
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS:
            files.append(p.resolve())
    return files

def infer_label_from_path(path: Path, class_candidates=None, root_dir: Path = None):
    """
    Verilen dosya yolundan, kök dizin altındaki hangi sınıf klasörüne ait olduğunu tespit et.
    Örn: .../breast/malignant/... -> label = 'malignant'
    """
    if class_candidates is None:
        class_candidates = ['benign', 'malignant']  # varsayılan, datasetine göre değiştir
    # path.parts'ta root_dir'den sonra gelen parçaları kontrol et
    if root_dir is None:
        root_dir = DATA_DIR
    for part in path.parts:
        if part.lower() in class_candidates:
            return part.lower()
    # fallback: parent folder name
    return path.parent.name.lower()

def build_filepath_label_lists(data_dir: Path):
    files = list_image_paths(data_dir)
    labels = []
    for f in files:
        lab = infer_label_from_path(f, root_dir=data_dir)
        labels.append(lab)
    # Map unique labels to integers
    classes = sorted(list(set(labels)))
    class_to_idx = {c: i for i, c in enumerate(classes)}
    numeric_labels = [class_to_idx[l] for l in labels]
    return files, numeric_labels, classes

# -------------------------------------------------------------------------
# Veri yollarını al, train/val/test split (80/10/10) yap
# -------------------------------------------------------------------------
all_files, all_labels, class_names = build_filepath_label_lists(DATA_DIR)
if len(all_files) == 0:
    raise SystemExit(f"Veri bulunamadı: {DATA_DIR}. Lütfen veri yolunu kontrol et.")

# Shuffle deterministically
rng = np.random.RandomState(SEED)
indices = np.arange(len(all_files))
rng.shuffle(indices)
all_files = [all_files[i] for i in indices]
all_labels = [all_labels[i] for i in indices]

# 80% train, 10% val, 10% test
train_paths, temp_paths, train_labels, temp_labels = train_test_split(
    all_files, all_labels, train_size=0.8, random_state=SEED, stratify=all_labels
)
val_paths, test_paths, val_labels, test_labels = train_test_split(
    temp_paths, temp_labels, test_size=0.5, random_state=SEED, stratify=temp_labels
)

print(f"Toplam görüntü: {len(all_files)}")
print(f"Train: {len(train_paths)}, Val: {len(val_paths)}, Test: {len(test_paths)}")
print(f"Sınıflar: {class_names}")

# -------------------------------------------------------------------------
# tf.data pipeline oluşturma
# -------------------------------------------------------------------------
def decode_and_resize(path, label):
    # path: tf.string
    img = tf.io.read_file(path)
    img = tf.image.decode_image(img, channels=3, expand_animations=False)
    img = tf.image.resize(img, IMG_SIZE)
    img = tf.cast(img, tf.float32) / 255.0
    return img, label

def prepare_dataset(file_paths, labels, batch_size=BATCH_SIZE, shuffle=False, augment=False):
    file_paths = [str(p) for p in file_paths]
    ds = tf.data.Dataset.from_tensor_slices((file_paths, labels))
    ds = ds.map(lambda p, l: tf.py_function(func=decode_and_resize, inp=[p, l], Tout=(tf.float32, tf.int32)), num_parallel_calls=AUTOTUNE)
    # After py_function, shapes may be unknown -> set shape
    def set_shapes(image, label):
        image.set_shape((IMG_SIZE[0], IMG_SIZE[1], 3))
        label.set_shape(())
        return image, label
    ds = ds.map(set_shapes, num_parallel_calls=AUTOTUNE)
    if shuffle:
        ds = ds.shuffle(buffer_size=1000, seed=SEED)
    if augment:
        # Keras augmentation layers (applied in graph)
        aug = keras.Sequential([
            layers.RandomFlip("horizontal_and_vertical"),
            layers.RandomRotation(0.15),
            layers.RandomZoom(0.15),
            layers.RandomContrast(0.15),
        ])
        ds = ds.map(lambda x, y: (aug(x, training=True), y), num_parallel_calls=AUTOTUNE)
    ds = ds.batch(batch_size).prefetch(AUTOTUNE)
    return ds

train_ds = prepare_dataset(train_paths, train_labels, shuffle=True, augment=True)
val_ds = prepare_dataset(val_paths, val_labels, shuffle=False, augment=False)
test_ds = prepare_dataset(test_paths, test_labels, shuffle=False, augment=False)

# -------------------------------------------------------------------------
# MODEL MİMARİSİ (esnek, create_model fonksiyonu)
# -------------------------------------------------------------------------
def create_model(input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3), base_filters=32, dropout_rate=0.4):
    model = keras.Sequential([
        layers.Input(shape=input_shape),

        layers.Conv2D(base_filters, 3, padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(),
        layers.Dropout(dropout_rate),

        layers.Conv2D(base_filters*2, 3, padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(),
        layers.Dropout(dropout_rate),

        layers.Conv2D(base_filters*4, 3, padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(),
        layers.Dropout(dropout_rate),

        layers.Flatten(),
        layers.Dense(256, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(1, activation='sigmoid')  # binary
    ])
    return model

# -------------------------------------------------------------------------
# Keras Tuner ile hiperparam arama (opsiyonel)
# -------------------------------------------------------------------------
if USE_KERAS_TUNER:
    try:
        import keras_tuner as kt
    except Exception as e:
        print("Keras Tuner yüklü değil. 'pip install keras-tuner' ile kur ve tekrar çalıştır.")
        raise

    def tuner_builder(hp):
        base_filters = hp.Choice("base_filters", [16, 32, 48])
        dropout_rate = hp.Float("dropout", 0.2, 0.5, step=0.1)
        lr = hp.Choice("learning_rate", [1e-2, 1e-3, 1e-4])
        model = create_model(base_filters=base_filters, dropout_rate=dropout_rate)
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=lr),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        return model

    tuner = kt.RandomSearch(
        tuner_builder,
        objective='val_accuracy',
        max_trials=6,
        executions_per_trial=1,
        directory='tuner_results',
        project_name='breakhis_tuner'
    )

    print("\n--- Keras Tuner araması başlıyor (kısa örnek: hızlı çalışması için epochs=5) ---")
    tuner.search(train_ds, validation_data=val_ds, epochs=5)
    best_hp = tuner.get_best_hyperparameters(num_trials=1)[0]
    print("En iyi hiperparametreler:")
    print(best_hp.values)
    # En iyi model
    model = tuner.get_best_models(num_models=1)[0]
else:
    # Default model
    model = create_model()

# -------------------------------------------------------------------------
# Compile model (eğer tuner kullanıldıysa model zaten compile edildi)
# -------------------------------------------------------------------------
if not USE_KERAS_TUNER:
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-4),
        loss='binary_crossentropy',
        metrics=['accuracy', keras.metrics.Precision(name='precision'), keras.metrics.Recall(name='recall'), keras.metrics.AUC(name='auc')]
    )

print("\nModel Özeti:")
model.summary()

# -------------------------------------------------------------------------
# Callback'ler
# -------------------------------------------------------------------------
checkpoint_path = OUTPUT_DIR / "best_model.h5"
callbacks = [
    keras.callbacks.ModelCheckpoint(str(checkpoint_path), monitor='val_accuracy', save_best_only=True, verbose=1),
    keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=1),
    keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=1e-7, verbose=1),
    keras.callbacks.CSVLogger(str(OUTPUT_DIR / "training_log.csv"))
]

# TensorBoard (isteğe bağlı)
tb_logdir = str(OUTPUT_DIR / "tensorboard")
callbacks.append(keras.callbacks.TensorBoard(log_dir=tb_logdir))

# -------------------------------------------------------------------------
# Model eğitimi
# -------------------------------------------------------------------------
EPOCHS = 30
history = model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS, callbacks=callbacks)

# -------------------------------------------------------------------------
# Değerlendirme: test seti
# -------------------------------------------------------------------------
print("\nTest seti üzerinde değerlendirme:")
results = model.evaluate(test_ds, verbose=1)
print(dict(zip(model.metrics_names, results)))

# Eğitim geçmişi CSV'ye kaydet
hist_df = pd.DataFrame(history.history)
hist_df.to_csv(OUTPUT_DIR / "history.csv", index=False)

# -------------------------------------------------------------------------
# Grafikler: Accuracy/Loss
# -------------------------------------------------------------------------
plt.figure(figsize=(12,5))
plt.subplot(1,2,1)
plt.plot(history.history.get('accuracy', []), label='train_acc')
plt.plot(history.history.get('val_accuracy', []), label='val_acc')
plt.title("Doğruluk (Accuracy)")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()

plt.subplot(1,2,2)
plt.plot(history.history.get('loss', []), label='train_loss')
plt.plot(history.history.get('val_loss', []), label='val_loss')
plt.title("Kayıp (Loss)")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "accuracy_loss.png")
plt.show()

# -------------------------------------------------------------------------
# Confusion matrix & Classification report (test set)
# -------------------------------------------------------------------------
# Test setten tüm görüntüleri alıp tahmin al
y_true = []
y_pred = []
y_prob = []

for x_batch, y_batch in test_ds:
    preds = model.predict(x_batch)
    y_prob.extend(preds.flatten().tolist())
    y_pred.extend((preds.flatten() > 0.5).astype(int).tolist())
    y_true.extend(y_batch.numpy().tolist())

print("\nClassification Report:")
print(classification_report(y_true, y_pred, target_names=class_names))

cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
plt.title("Confusion Matrix (Test)")
plt.xlabel("Tahmin Edilen")
plt.ylabel("Gerçek")
plt.savefig(OUTPUT_DIR / "confusion_matrix.png")
plt.show()

# ROC - AUC
fpr, tpr, _ = roc_curve(y_true, y_prob)
roc_auc = auc(fpr, tpr)
plt.figure(figsize=(6,6))
plt.plot(fpr, tpr, lw=2, label=f"AUC = {roc_auc:.3f}")
plt.plot([0,1],[0,1], linestyle="--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Eğrisi (Test)")
plt.legend()
plt.savefig(OUTPUT_DIR / "roc_auc.png")
plt.show()

# -------------------------------------------------------------------------
# Modeli kaydet (hem HDF5 hem SavedModel)
# -------------------------------------------------------------------------
final_h5 = OUTPUT_DIR / "breakhis_final_model.h5"
final_tf = OUTPUT_DIR / "breakhis_final_savedmodel"
model.save(final_h5)
model.save(final_tf)  # SavedModel format
print(f"Model kaydedildi: {final_h5} ve {final_tf}")

print("\nEğitim ve değerlendirme tamamlandı.")
