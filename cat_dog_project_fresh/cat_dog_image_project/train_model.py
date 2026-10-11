"""
Train a Cat vs Dog image classifier using MobileNetV2 transfer learning.

Expected dataset structure:
data/
  cats/
    cat.1.jpg
    ...
  dogs/
    dog.1.jpg
    ...

Run:
  python train_model.py
"""
from pathlib import Path
import json
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

DATA_DIR = Path("data_clean")
MODEL_DIR = Path("artifacts")
MODEL_PATH = MODEL_DIR / "cat_dog_model.keras"
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
SEED = 42
EPOCHS_HEAD = 5
EPOCHS_FINE_TUNE = 3

def main():
    if not DATA_DIR.exists():
        raise FileNotFoundError(
            "Dataset folder 'data/' was not found. Add data/cats/ and data/dogs/ first. "
            "See README.md for instructions."
        )

    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    train_ds = keras.utils.image_dataset_from_directory(
        DATA_DIR,
        validation_split=0.2,
        subset="training",
        seed=SEED,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        label_mode="binary",
    )
    val_ds = keras.utils.image_dataset_from_directory(
        DATA_DIR,
        validation_split=0.2,
        subset="validation",
        seed=SEED,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        label_mode="binary",
    )

    class_names = train_ds.class_names
    print("Class order:", class_names)
    if len(class_names) != 2:
        raise ValueError(f"Expected exactly 2 folders/classes, found: {class_names}")

    autotune = tf.data.AUTOTUNE
    train_ds = train_ds.cache().shuffle(1000, seed=SEED).prefetch(autotune)
    val_ds = val_ds.cache().prefetch(autotune)

    data_augmentation = keras.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.08),
        layers.RandomZoom(0.1),
    ], name="data_augmentation")

    base_model = keras.applications.MobileNetV2(
        input_shape=(*IMG_SIZE, 3),
        include_top=False,
        weights="imagenet",
    )
    base_model.trainable = False

    inputs = keras.Input(shape=(*IMG_SIZE, 3))
    x = data_augmentation(inputs)
    x = keras.applications.mobilenet_v2.preprocess_input(x)
    x = base_model(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.25)(x)
    outputs = layers.Dense(1, activation="sigmoid")(x)
    model = keras.Model(inputs, outputs)

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )

    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=2, restore_best_weights=True
        ),
        keras.callbacks.ModelCheckpoint(
            filepath=str(MODEL_PATH), monitor="val_loss",
            save_best_only=True
        ),
    ]

    print("\nStage 1: training the classification head")
    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS_HEAD,
        callbacks=callbacks,
    )

    print("\nStage 2: fine-tuning the last MobileNetV2 layers")
    base_model.trainable = True
    for layer in base_model.layers[:-30]:
        layer.trainable = False

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-5),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS_FINE_TUNE,
        callbacks=callbacks,
    )

    # Reload the best saved model, then save metadata for the app.
    model = keras.models.load_model(MODEL_PATH)
    loss, accuracy = model.evaluate(val_ds, verbose=0)
    metadata = {
        "class_names": class_names,
        "image_size": list(IMG_SIZE),
        "validation_loss": float(loss),
        "validation_accuracy": float(accuracy),
    }
    (MODEL_DIR / "metadata.json").write_text(json.dumps(metadata, indent=2))
    print(f"\nSaved model to: {MODEL_PATH}")
    print(f"Validation accuracy: {accuracy:.4f}")
    print("Class mapping:", dict(enumerate(class_names)))

if __name__ == "__main__":
    main()
