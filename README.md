# 🌿 Plant Disease Detection

![Plant Image](https://cdn-images-1.medium.com/max/1200/1*FswlF4lZPQ4kT_gkybacZw.jpeg)

## 📌 Introduction

Getting affected by a disease is very common in plants due to various factors such as fertilizers, cultural practices followed, environmental conditions, etc. These diseases hurt agricultural yield and eventually the economy based on it.

Any technique or method to overcome this problem and get a warning before the plants are infected would help farmers cultivate more efficiently—both qualitatively and quantitatively. Therefore, disease detection in plants plays a critical role in agriculture.

## 📂 Dataset — PlantVillage

This project uses the widely popular **[PlantVillage Dataset](https://www.kaggle.com/datasets/abdallahalidev/plantvillage-dataset)**, available on Kaggle.

The dataset consists of approximately 54,305 images of plant leaves collected under controlled environmental conditions. These images include leaves from 14 plant species spanning 38 disease classes plus one "healthy" class for each species. The species represented are:

- **Apple**: Scab, Black Rot, Cedar Rust, Healthy  
- **Blueberry**: Healthy  
- **Cherry**: Powdery Mildew, Healthy  
- **Corn**: Gray Leaf Spot, Common Rust, Northern Leaf Blight, Healthy  
- **Grape**: Black Rot, Black Measles, Leaf Blight, Healthy  
- **Orange**: Huanglongbing  
- **Peach**: Bacterial Spot, Healthy  
- **Bell Pepper**: Bacterial Spot, Healthy  
- **Potato**: Early Blight, Late Blight, Healthy  
- **Raspberry**: Healthy  
- **Soybean**: Healthy  
- **Squash**: Powdery Mildew  
- **Strawberry**: Leaf Scorch, Healthy  
- **Tomato**: Bacterial Spot, Early Blight, Late Blight, Leaf Mold, Septoria Leaf Spot, Two-Spotted Spider Mite, Target Spot, Mosaic Virus, Yellow Leaf Curl Virus, Healthy  


Due to limited local compute power, using Google Colab with a TPU backend enables quick and effective model training and experimentation.

## 🚀 Google Colab Notebook

Run the end-to-end workflow (data loading, preprocessing, training, evaluation, and inference) in Colab:

👉 **[Open Colab Notebook](https://colab.research.google.com/drive/1d3mVcoinztULLif3DzTMcbcA7KGha14W?usp=sharing)**


Feel free to open the notebook to view or run the code directly—training, evaluation, and all included insights are there for exploration.

## 🧰 What’s Inside (at a glance)
- Dataset preparation & augmentation  
- Transfer learning with CNN backbones (e.g., MobileNet/ResNet)  
- Training & validation loops  
- Metrics (accuracy, confusion matrix)  
- Inference on custom images
