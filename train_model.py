import tensorflow as tf
import numpy as np
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from sklearn.utils.class_weight import compute_class_weight

DATASET_DIR = "dataset"
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS_HEAD = 15       # tahap 1: cuma latih classifier head
EPOCHS_FINETUNE = 10   # tahap 2: fine-tune sebagian layer MobileNetV2

# 1. Data generator dengan augmentasi + split train/val otomatis
datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=25,
    width_shift_range=0.15,
    height_shift_range=0.15,
    zoom_range=0.15,
    shear_range=0.1,
    horizontal_flip=True,
    brightness_range=[0.6, 1.4],
    validation_split=0.2
)

# Validation TIDAK pakai augmentasi (cuma rescale), biar evaluasi lebih representatif
val_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2
)

train_gen = datagen.flow_from_directory(
    DATASET_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='training',
    shuffle=True
)

val_gen = val_datagen.flow_from_directory(
    DATASET_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation',
    shuffle=False
)

print("Kelas terdeteksi:", train_gen.class_indices)
NUM_CLASSES = len(train_gen.class_indices)

# ===== Hitung class weight otomatis (BARU) =====
# Karena kertas & plastik sekarang jauh lebih banyak, ini kasih bobot lebih
# ke kelas yang datanya lebih sedikit (misal kaleng), biar model tidak bias
labels = train_gen.classes
class_weights_array = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(labels),
    y=labels
)
class_weights = dict(enumerate(class_weights_array))
print("Class weights (buat imbangi jumlah data tiap kelas):", class_weights)

# 2. Bangun model dari MobileNetV2 (transfer learning)
base_model = MobileNetV2(input_shape=(224, 224, 3), include_top=False, weights='imagenet')
base_model.trainable = False

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dropout(0.4)(x)
x = Dense(128, activation='relu')(x)
x = Dropout(0.2)(x)
output = Dense(NUM_CLASSES, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=output)
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
              loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

# ===== Callbacks (BARU) =====
checkpoint = ModelCheckpoint(
    "waste_classifier_best.keras",
    monitor='val_accuracy',
    save_best_only=True,
    mode='max',
    verbose=1
)
early_stop = EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True,
    verbose=1
)
reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=3,
    min_lr=1e-6,
    verbose=1
)

# ===== TAHAP 1: latih classifier head saja (base model freeze) =====
print("\n===== TAHAP 1: Training classifier head =====\n")
history_head = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=EPOCHS_HEAD,
    class_weight=class_weights,
    callbacks=[checkpoint, early_stop, reduce_lr]
)

# ===== TAHAP 2: Fine-tuning, unfreeze sebagian layer akhir MobileNetV2 (BARU) =====
print("\n===== TAHAP 2: Fine-tuning sebagian layer MobileNetV2 =====\n")
base_model.trainable = True

# Cuma unfreeze beberapa layer terakhir, sisanya tetap freeze
# supaya tidak merusak fitur low-level yang sudah bagus dari ImageNet
FINE_TUNE_AT = len(base_model.layers) - 30
for layer in base_model.layers[:FINE_TUNE_AT]:
    layer.trainable = False

# Learning rate lebih kecil buat fine-tuning, biar tidak merusak weight yang sudah bagus
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
              loss='categorical_crossentropy', metrics=['accuracy'])

history_finetune = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=EPOCHS_FINETUNE,
    class_weight=class_weights,
    callbacks=[checkpoint, early_stop, reduce_lr]
)

# 4. Simpan model final (native Keras format)
model.save("waste_classifier_final.keras")
print("\nModel final tersimpan sebagai waste_classifier_final.keras")
print("Model terbaik (val_accuracy tertinggi) tersimpan sebagai waste_classifier_best.keras")

# ===== Evaluasi akhir =====
final_val_loss, final_val_acc = model.evaluate(val_gen)
print(f"\nFinal validation accuracy: {final_val_acc*100:.2f}%")
print(f"Final validation loss: {final_val_loss:.4f}")