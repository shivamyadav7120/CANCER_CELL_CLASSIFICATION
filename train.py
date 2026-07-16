import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten
from tensorflow.keras.layers import Dense, Dropout, Input

# Dataset Paths
TRAIN_PATH = "dataset/train"
VALID_PATH = "dataset/valid"

# Image Generator
train_data = ImageDataGenerator(
    rescale=1./255
).flow_from_directory(
    TRAIN_PATH,
    target_size=(224, 224),
    batch_size=32,
    class_mode="categorical"
)

valid_data = ImageDataGenerator(
    rescale=1./255
).flow_from_directory(
    VALID_PATH,
    target_size=(224, 224),
    batch_size=32,
    class_mode="categorical"
)

print("Classes:")
print(train_data.class_indices)

# CNN Model
model = Sequential([

    Input(shape=(224,224,3)),

    Conv2D(32, (3,3), activation="relu"),
    MaxPooling2D(2,2),

    Conv2D(64, (3,3), activation="relu"),
    MaxPooling2D(2,2),

    Conv2D(128, (3,3), activation="relu"),
    MaxPooling2D(2,2),

    Flatten(),

    Dense(128, activation="relu"),
    Dropout(0.5),

    Dense(7, activation="softmax")
])

# Compile Model
model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

# Train Model
model.fit(
    train_data,
    validation_data=valid_data,
    epochs=10
)

# Save Model
model.save("cancer_model.keras")

print("Model Saved Successfully!")