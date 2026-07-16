import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing import image

model = tf.keras.models.load_model("cancer_model.keras")

classes = [
    "akiec",
    "bcc",
    "bkl",
    "df",
    "mel",
    "nv",
    "vasc"
]

img_path = "test.jpg"

img = image.load_img(img_path, target_size=(224, 224))
img = image.img_to_array(img)
img = np.expand_dims(img, axis=0)
img = img / 255.0

prediction = model.predict(img)

index = np.argmax(prediction)

print("Predicted Class :", classes[index])
print("Confidence :", np.max(prediction) * 100, "%")

top3 = np.argsort(prediction[0])[-3:][::-1]

print("\nTop-3 Predictions")

for i in top3:
    print(classes[i], ":", round(prediction[0][i] * 100, 2), "%")

confidence = prediction[0] * 100

plt.bar(classes, confidence)
plt.title("Prediction Confidence")
plt.xlabel("Classes")
plt.ylabel("Confidence (%)")
plt.show()