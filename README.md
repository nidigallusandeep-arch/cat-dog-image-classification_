# Cat vs Dog Image Classification

An end-to-end image classification project using **TensorFlow/Keras, MobileNetV2 transfer learning, and Streamlit**.

## What it does
- Loads cat and dog images from folders.
- Resizes images to 224 × 224.
- Applies light data augmentation.
- Uses a pretrained MobileNetV2 feature extractor.
- Trains a binary classifier and fine-tunes the final layers.
- Evaluates on a held-out validation split.
- Provides a Streamlit app for image uploads.

> Validation results depend on your dataset and training run. Do not claim a particular accuracy before measuring it.

## 1. Setup (Mac)

Use Python 3.10 or 3.11 for a compatible TensorFlow environment.

```bash
mkdir cat-dog-image-project
cd cat-dog-image-project
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If you downloaded this project ZIP, run the commands from the extracted project folder.

## 2. Add the dataset

Create this structure:

```text
data/
├── cats/
│   ├── cat.1.jpg
│   ├── cat.2.jpg
│   └── ...
└── dogs/
    ├── dog.1.jpg
    ├── dog.2.jpg
    └── ...
```

Use a public Cats vs Dogs dataset from Microsoft/Kaggle or another source whose license permits your use. Download it yourself from its official dataset page, review the terms, and copy images into the two folders. Keep a reasonable number of images per class and remove broken/non-image files. The training script automatically reserves 20% for validation.

## 3. Train the model

```bash
python train_model.py
```

The first run downloads pretrained MobileNetV2 weights. Training can take time and uses memory. The best model and metrics are saved in `artifacts/`.

## 4. Run the app

```bash
streamlit run app.py
```

Open the local URL printed in the terminal and upload a JPG, PNG, or WEBP image.

## 5. Suggested GitHub structure

```text
cat-dog-image-project/
├── app.py
├── train_model.py
├── requirements.txt
├── README.md
├── .gitignore
├── data/                 # local only; not committed
└── artifacts/            # trained model; not committed by default
```

## Evaluation and improvement
- Record validation accuracy and loss from your actual run.
- Test with new images not used in training.
- Inspect false positives/false negatives and a confusion matrix.
- Keep training and validation data separate to reduce data leakage.
- Try more training images, careful augmentation, and compare against a simple CNN baseline.
- Avoid describing the model's score as a guaranteed probability.

## Interview explanation (simple English)
"I built a Cat vs Dog image classification application using TensorFlow and MobileNetV2 transfer learning. I resized and augmented the images, froze the pretrained feature extractor while training a binary classification head, then fine-tuned the last layers with a smaller learning rate. I evaluated the model on a validation split and deployed it with Streamlit so users can upload an image and get a prediction."

## Important
This is a learning project, not a production-grade animal recognition system. Results depend on dataset quality, class balance, and images the model has not seen before.
