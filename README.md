# 🐾 FurSight: Cat & Dog Disease Detection

FurSight is a real-time application for detecting skin-related diseases in cats and dogs from images. It uses a combination of **CNN feature extraction** and a **Graph Neural Network (GNN)** classifier for accurate predictions. Built with **PyTorch**, **Torch Geometric**, and **Streamlit**, FurSight allows pet owners and veterinarians to quickly analyze affected areas.

---

## 🔹 Key Features

- Upload an image of your pet’s affected skin or fur.  
- Predicts **3 main skin-related classes**:
  - **Demodicosis** (mange)  
  - **Dermatitis**  
  - **Healthy** skin  
- Works on images of the affected area (not necessarily the whole pet).  
- Real-time predictions through an interactive **Streamlit** interface.  
- Optional visualization of predictions via heatmaps (future updates).  

---

## 🔹 Dataset

- Original dataset contains **28 classes** (various skin and non-skin diseases).  
- For this project, we focused on **3 classes** related to skin problems.  
- Combined **two different datasets** to increase training diversity, though the dataset size is still limited:  
  - [Pet Disease Images](https://www.kaggle.com/datasets/smadive/pet-disease-images?resource=download)  
  - [Dogs Skin Diseases Image Dataset](https://www.kaggle.com/datasets/youssefmohmmed/dogs-skin-diseases-image-dataset?)  
- The app **cannot replace a veterinarian**, but it may help for quick screening or preliminary analysis.  
- The dataset includes other disease types (not only skin issues), highlighting that FurSight is currently optimized for skin conditions.

---

## 🔹 Demo

- **Main Page:**  
![Main Page](./fur.JPG)

- **1st Case:**  
![Case 1](./demo.JPG)

- **2nd Case:**  
![Case 2](./healthy_cat.JPG)

- **3rd Case:**  
![Case 3](./dog_derm.JPG)

---


## 🔹 Installation



```bash
pip install -r requirements.txt
