# AI-Generated Image Detector

**An end-to-end computer vision tutorial project:** a binary image classifier trained with transfer learning, evaluated on held-out CIFAKE images, and deployed as an interactive Gradio web app.

**[Live demo]**: Add your Render `.onrender.com` URL here after verifying image upload and prediction.

**Repository:** https://github.com/pobtanin24-prog/ai-image-detector

> **Scope:** This is an experimental classifier, not a forensic tool. It is not validated for all modern image generators, edited images, or screenshots.

## What I built

- Prepared a balanced subset of the CIFAKE dataset and kept the test split separate from training.
- Used pretrained **ResNet18** (ImageNet weights) as a feature extractor.
- Froze the pretrained layers and trained a new 2-class linear classification head.
- Evaluated the saved model on **1,000 held-out images**.
- Built a **Gradio** interface for image upload and AI/real predictions.
- Deployed the application to **Render**, with a CPU-only PyTorch environment.

### Pipeline

```text
CIFAKE (REAL / FAKE)
         |
         v
Select 4,000 training + 1,000 test images
         |
         v
Resize to 224 x 224 + ImageNet normalization
         |
         v
Pretrained ResNet18 (frozen backbone)
         |
         v
Train 2-class linear head
         |
         v
Evaluate on held-out CIFAKE images
         |
         v
Gradio web app -> Render deployment -> QR demo
```

## Dataset and method

| Item | Configuration |
| --- | --- |
| Source | [CIFAKE](https://www.kaggle.com/datasets/birdy654/cifake-real-and-ai-generated-synthetic-images) |
| Full dataset | 60,000 REAL and 60,000 FAKE images |
| REAL images | Derived from CIFAR-10 |
| FAKE images | Generated with Stable Diffusion 1.4 |
| Subset used | 4,000 train (2,000/class); 1,000 test (500/class) |
| Model | ResNet18 with ImageNet pretrained weights |
| Trainable part | Final linear classification layer only |
| Training | 3 epochs, batch size 32, Adam, learning rate 0.001 |
| Compute | CPU |
| Class order | 0 = FAKE; 1 = REAL |

No test images were used to optimize the model in the training script. The classifier uses softmax outputs as **model scores**, not calibrated probabilities.

## Results

**Recorded from one local training run:**

| Epoch | Training loss | Training accuracy |
| --- | ---: | ---: |
| 1 | 0.5354 | 72.25% |
| 2 | 0.4018 | 82.10% |
| 3 | 0.3751 | 83.60% |

**Held-out CIFAKE subset test accuracy: 85.40% (854/1,000 images).**

This number measures performance only on the selected CIFAKE test subset. It does **not** establish 85.4% accuracy on arbitrary AI-generated images.

For a more complete evaluation, run `evaluate.py` locally to generate per-class precision, recall, F1, and a confusion matrix. Those additional metrics are not yet reported here because they have not been supplied by an evaluation run.

## Run locally

1. Create and activate a Python environment.
2. Install dependencies: `pip install -r requirements.txt`.
3. Confirm `ai_detector.pth` is present at the project root.
4. Run `python app.py` and open the local URL shown in the terminal.

The model checkpoint is included for inference. Training data is intentionally excluded from GitHub because of its size.

### Reproduce / further evaluate

1. Download CIFAKE from the dataset link above and extract it so that `cifake-full/train/FAKE`, `cifake-full/train/REAL`, `cifake-full/test/FAKE`, and `cifake-full/test/REAL` exist.
2. Run `python prepare_data.py` to create the small training and test subsets.
3. Run `python train.py` to train a model. Results may vary because the original training script does not fix all random seeds.
4. In an environment with `scikit-learn` and `matplotlib` installed, run `python evaluate.py` to evaluate `ai_detector.pth` on the local `data/test` split. It writes `results/evaluation.json` and `results/confusion_matrix.png`.

**Important:** For comparison against the originally reported 85.40%, evaluate the **same original local `data/test` subset** with the **same original checkpoint**. Recreating or replacing the subset can change the measured accuracy.

## Engineering decisions

- **Transfer learning over training from scratch:** reduced training cost enough to run on a laptop CPU.
- **Small balanced dataset over all 120,000 images:** suitable for learning the ML workflow and building a working demo quickly.
- **CPU-only deployment:** avoids GPU/CUDA dependencies on a CPU web host.
- **Simple upload UI:** prioritizes a usable proof-of-work over extra features that would not improve model validity.

## Limitations and next experiment

- CIFAKE images are originally 32 x 32 pixels; resizing them to 224 x 224 does not create new image detail.
- The FAKE class comes from Stable Diffusion 1.4, so generalization to unseen generators is unknown.
- The model may learn dataset-specific shortcuts instead of broadly generalizable indicators of synthetic imagery.
- Softmax confidence does not imply calibrated real-world certainty.
- The model is not designed to detect partially edited or manipulated real photographs.

**Next useful experiment:** Test the existing checkpoint on an independently sourced set of real photographs and images from a different generator, without retraining, and report the results separately from CIFAKE accuracy.

## Evidence checklist

- [x] Public source code and training pipeline
- [x] Saved model checkpoint
- [x] Recorded training log and held-out test accuracy
- [x] Working image-upload demo (confirm the deployed link before sharing)
- [ ] Screenshot of deployed app with a completed prediction
- [ ] Per-class precision / recall / F1 and confusion matrix from `evaluate.py`
- [ ] Cross-generator holdout experiment (optional)

## Credits

This is a **learning project** using established datasets and pretrained architectures, not a claim of a newly invented AI detection method.

- Dataset: Bird, J. J. and Lotfi, A. (2024). *CIFAKE: Image Classification and Explainable Identification of AI-Generated Synthetic Images*, IEEE Access. [Dataset repository](https://github.com/jordan-bird/CIFAKE-Real-and-AI-Generated-Synthetic-Images)
- Real-image source: Krizhevsky, A. and Hinton, G. (2009). *Learning multiple layers of features from tiny images* (CIFAR-10).
- Model: ResNet18 via [TorchVision](https://pytorch.org/vision/stable/models/generated/torchvision.models.resnet18.html).
- App: [Gradio](https://www.gradio.app/).
