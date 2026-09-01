import tensorflow as tf

# Pakai model TERBAIK hasil training (5 kelas), bukan yang lama
model = tf.keras.models.load_model("waste_classifier_best.keras")

converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_model = converter.convert()

with open("waste_classifier.tflite", "wb") as f:
    f.write(tflite_model)

print("Model TFLite tersimpan.")