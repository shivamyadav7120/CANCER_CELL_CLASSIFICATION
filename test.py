import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

MODEL_PATH = "cancer_model.keras"
TEST_DIR = "dataset/test"
RESULTS_DIR = "results"

os.makedirs(RESULTS_DIR, exist_ok=True)

class_names = [
    "akiec",
    "bcc",
    "bkl",
    "df",
    "mel",
    "nv",
    "vasc"
]

model = load_model(MODEL_PATH)

test_datagen = ImageDataGenerator(rescale=1.0 / 255)

test_generator = test_datagen.flow_from_directory(
    TEST_DIR,
    target_size=(224, 224),
    batch_size=32,
    class_mode="categorical",
    shuffle=False
)

loss, accuracy = model.evaluate(test_generator)

print("\nTest Loss :", loss)
print("Test Accuracy :", accuracy * 100, "%")

predictions = model.predict(test_generator)

y_pred = np.argmax(predictions, axis=1)
y_true = test_generator.classes

acc = accuracy_score(y_true, y_pred)

print("\nAccuracy Score :", acc * 100, "%")

report = classification_report(
    y_true,
    y_pred,
    target_names=class_names
)

print("\nClassification Report\n")
print(report)

with open(os.path.join(RESULTS_DIR, "classification_report.txt"), "w") as f:
    f.write(report)

cm = confusion_matrix(y_true, y_pred)

plt.figure(figsize=(8, 7))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=class_names,
    yticklabels=class_names
)

plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")

plt.tight_layout()

plt.savefig(os.path.join(RESULTS_DIR, "confusion_matrix.png"))
plt.show()

with open(os.path.join(RESULTS_DIR, "test_accuracy.txt"), "w") as f:
    f.write(f"Test Accuracy: {accuracy * 100:.2f}%\n")
    f.write(f"Test Loss: {loss:.4f}\n")

filenames = test_generator.filenames

print("\nSample Predictions\n")

for i in range(min(10, len(filenames))):
    print("Image :", filenames[i])
    print("Actual :", class_names[y_true[i]])
    print("Predicted :", class_names[y_pred[i]])
    print("-" * 40)

print("\nTesting Completed Successfully!")