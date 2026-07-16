import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Image Size
IMG_SIZE = 224

# Dataset Path
DATASET = "dataset/train"

# Class Names
classes = [
    "akiec",
    "bcc",
    "bkl",
    "df",
    "mel",
    "nv",
    "vasc"
]

# Lists for Images and Labels
X = []
y = []

# Read Images
for label, cls in enumerate(classes):

    folder = os.path.join(DATASET, cls)

    if not os.path.exists(folder):
        print(f"{folder} not found")
        continue

    for file in os.listdir(folder):

        img_path = os.path.join(folder, file)

        img = cv2.imread(img_path)

        if img is None:
            continue

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        img = img / 255.0

        X.append(img)
        y.append(label)

# Convert into NumPy Arrays
X = np.array(X, dtype=np.float32)
y = np.array(y)

# Display Information
print("Total Images :", len(X))
print("Image Shape  :", X.shape)
print("Label Shape  :", y.shape)

print("\nClasses")
for i, cls in enumerate(classes):
    print(i, ":", cls)

# Show Sample Image
index = 50

plt.figure(figsize=(5,5))
plt.imshow(X[index])
plt.title(classes[y[index]])
plt.axis("off")
plt.show()