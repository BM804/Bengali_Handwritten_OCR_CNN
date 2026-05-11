import os
import math
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from PIL import Image
model_path = "model/final_bangla_cnn_model.keras"

model = tf.keras.models.load_model(model_path)

print("MODEL LOADED SUCCESSFULLY")
test_folder = "test_images"

image_files = sorted([
    f for f in os.listdir(test_folder)
    if f.endswith((".png", ".jpg", ".jpeg"))
])

print("TOTAL IMAGES :", len(image_files))

cols = 4
rows = math.ceil(len(image_files) / cols)

fig, axes = plt.subplots(
    rows,
    cols,
    figsize=(16, rows * 4)
)

fig.suptitle(
    "Bangla Character Prediction",
    fontsize=20,
    fontweight='bold'
)

axes = axes.flatten()
for idx, file_name in enumerate(image_files):

    image_path = os.path.join(test_folder, file_name)

    # LOAD IMAGE

    img = Image.open(image_path).convert("L")

    img = img.resize((32, 32))

    img_array = np.array(img)

    # PREPROCESS

    input_image = img_array / 255.0

    input_image = input_image.reshape(1, 32, 32, 1)

    # PREDICTION

    prediction = model.predict(input_image, verbose=0)

    predicted_label = np.argmax(prediction)

    confidence = np.max(prediction) * 100
    axes[idx].imshow(img_array, cmap='gray')

    axes[idx].set_title(
        f"Prediction : {predicted_label}\n"
        f"Confidence : {confidence:.2f}%",
        fontsize=11,
        fontweight='bold',
        pad=10
    )

    axes[idx].axis("off")
for j in range(idx + 1, len(axes)):
    axes[j].axis("off")
plt.tight_layout(rect=[0, 0, 1, 0.96])

plt.show()