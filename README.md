# DCGAN Face Generation using PyTorch

## Overview

This project implements a **Deep Convolutional Generative Adversarial Network (DCGAN)** using **PyTorch** to generate realistic human face images from random noise.

Unlike a traditional GAN that uses fully connected layers, DCGAN leverages **Convolutional Neural Networks (CNNs)** to learn spatial features, resulting in significantly better image quality and more stable training.

The model is trained on the **CelebA Dataset** and generates **64×64 RGB face images**.

---

## What is DCGAN?

DCGAN stands for:

**Deep Convolutional Generative Adversarial Network**

It consists of two networks:

### Generator (G)

* Takes random noise as input.
* Generates realistic face images.

### Discriminator (D)

* Receives real and generated images.
* Predicts whether an image is real or fake.

Both networks compete against each other during training, enabling the Generator to create increasingly realistic images.

---

## Dataset

### CelebA Dataset

* 202,599 celebrity face images
* RGB Images
* Large-scale face dataset
* Images resized to **64 × 64**

Dataset Structure:

```text
img_align_celeba/
├── 000001.jpg
├── 000002.jpg
├── 000003.jpg
...
```

---

## Data Preprocessing

```python
transforms.Compose([
    transforms.CenterCrop(178),
    transforms.Resize(64),
    transforms.ToTensor(),
    transforms.Normalize(
        (0.5,0.5,0.5),
        (0.5,0.5,0.5)
    )
])
```

### Steps

* Center Crop
* Resize to 64×64
* Convert to Tensor
* Normalize pixel values to [-1,1]

---

## Generator Architecture

```text
Noise Vector
(100 × 1 × 1)
        ↓
ConvTranspose2d
        ↓
BatchNorm2d
        ↓
ReLU
        ↓
ConvTranspose2d
        ↓
BatchNorm2d
        ↓
ReLU
        ↓
ConvTranspose2d
        ↓
BatchNorm2d
        ↓
ReLU
        ↓
ConvTranspose2d
        ↓
BatchNorm2d
        ↓
ReLU
        ↓
ConvTranspose2d
        ↓
Tanh
        ↓
Generated Image
(3 × 64 × 64)
```

---

## Discriminator Architecture

```text
Image
(3 × 64 × 64)
        ↓
Conv2d
        ↓
LeakyReLU
        ↓
Conv2d
        ↓
BatchNorm2d
        ↓
LeakyReLU
        ↓
Conv2d
        ↓
BatchNorm2d
        ↓
LeakyReLU
        ↓
Conv2d
        ↓
BatchNorm2d
        ↓
LeakyReLU
        ↓
Conv2d
        ↓
Sigmoid
        ↓
Real/Fake Probability
```

---

## Hyperparameters

```python
Batch Size       = 128
Epochs           = 10
Learning Rate    = 0.0002
Optimizer        = Adam
Beta1            = 0.5
Beta2            = 0.999
Latent Dimension = 100
Image Size       = 64 × 64
```

---

## Loss Function

```python
nn.BCELoss()
```

Binary Cross Entropy Loss is used for:

* Discriminator Loss
* Generator Loss

---

## Training Procedure

### Train Discriminator

```text
Real Images  → Label 1
Fake Images  → Label 0
```

Discriminator learns to identify fake images.

---

### Train Generator

```text
Fake Images → Label 1
```

Generator learns to fool the Discriminator.

---

## Technologies Used

* Python
* PyTorch
* Torchvision
* NumPy
* Matplotlib
* PIL

---

## Results

The Generator progressively improves and learns to generate realistic human face images.

Sample Outputs:

```text
Epoch 1
↓
Blurry Faces

Epoch 5
↓
Better Facial Features

Epoch 10
↓
More Realistic Faces
```

---

## Project Structure

```text
DCGAN-Face-Generation/
│
├── dcgan.ipynb
├── README.md
├── requirements.txt
│
├── outputs/
│   ├── epoch_1.png
│   ├── epoch_5.png
│   ├── epoch_10.png
│   └── generated_faces.png
│
└── dataset/
    └── img_align_celeba/
```

---

## Why DCGAN Over GAN?

| GAN                      | DCGAN                     |
| ------------------------ | ------------------------- |
| Fully Connected Layers   | Convolutional Layers      |
| Lower Image Quality      | Better Image Quality      |
| Less Stable Training     | More Stable Training      |
| Limited Feature Learning | Learns Spatial Features   |
| Basic Image Generation   | Realistic Face Generation |

---

## Future Improvements

* Train for more epochs
* Generate higher-resolution images
* Implement Conditional GAN (CGAN)
* Implement StyleGAN
* Add model checkpoint saving
* Deploy using Flask/FastAPI

---

## Key Learnings

* Generative Adversarial Networks
* Deep Convolutional Neural Networks
* Conv2d & ConvTranspose2d
* Batch Normalization
* Adversarial Training
* Image Generation with PyTorch

---

## Author

Built as part of my AI/ML learning journey while exploring Generative AI and Computer Vision using PyTorch.
