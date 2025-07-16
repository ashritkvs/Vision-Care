# 👁️ VisionCare AI: Diabetic Retinopathy Detection System

VisionCare AI is a deep learning-powered diagnostic tool for the early detection of diabetic retinopathy (DR) using retinal fundus images. Built with PyTorch and Streamlit, this end-to-end project enables real-time DR classification through a clean and interactive web interface.

---

## 🚀 Demo

📸 Upload a retinal fundus image and receive an instant prediction:

![App Screenshot](screenshots/prediction_demo.png)

---

## 🧠 Features

- 🧬 ResNet50 CNN model trained on APTOS 2019 dataset
- 📊 Achieves 65% macro F1 score on validation set
- 🧾 Streamlit-powered UI for real-time inference
- 🧠 Five-class classification: No DR, Mild, Moderate, Severe, Proliferative DR
- 🔒 Model weights excluded from repo to ensure clean deployment

---

## 🧰 Tech Stack

| Component     | Tools / Libraries               |
|---------------|----------------------------------|
| Model         | PyTorch, ResNet50               |
| Augmentation  | Albumentations                  |
| Web App       | Streamlit                       |
| Image Proc    | OpenCV, PIL                     |
| Deployment    | Streamlit Cloud (optional)      |

---




---

## ⚙️ Setup & Installation

bash
git clone https://github.com/ashritkvs/Vision-Care.git
cd Vision-Care
pip install -r requirements.txt
streamlit run app/main.py

---

## 📂 Dataset

This project uses the [APTOS 2019 Blindness Detection Dataset](https://www.kaggle.com/competitions/aptos2019-blindness-detection/data).

To train your own model:

1. Go to the Kaggle link above
2. Download:
   - `train.csv`
   - `test.csv`
   - `train_images/`
   - `test_images/`



