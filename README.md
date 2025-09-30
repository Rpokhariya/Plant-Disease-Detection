# Live Demo : [**➡️ Click here for the Live Demo**]((https://huggingface.co/spaces/Rpokhariya/plant-disease-detection-application)


## Plant  Disease Detection

### 📖 Project Description

This project is an intelligent image classification system engineered to accurately diagnose **39 distinct types of plant diseases** directly from leaf images. The core of this system is a **Convolutional Neural Network (CNN)**, a deep learning model meticulously trained to achieve **86% accuracy** on the validation set.

To make this powerful diagnostic tool accessible and easy to use, an interactive web application was developed using **Streamlit**. This application allows users to simply upload an image of a plant leaf and receive an instant diagnosis. A built-in preprocessing tool automatically prepares the uploaded image, ensuring it is in the optimal format for the model's analysis.

---

### ✨ Features

-   **High Accuracy:** Leverages a CNN model to identify 39 different plant diseases with 86% validation accuracy.
-   **User-Friendly Interface:** An interactive and intuitive web application built with Streamlit for a seamless user experience.
-   **Instant Diagnosis:** Provides real-time disease identification upon image upload.
-   **Automated Image Preprocessing:** Includes a built-in tool to automatically resize and prepare user-uploaded images for the model.

---

### 🛠️ Technologies Used

-   **Backend & Model:** Python
-   **Deep Learning:** TensorFlow, Keras
-   **Web Framework:** Streamlit

---

### 🚀 How to Use

1.  **Navigate to the Live Demo:** Click the link in the [Live Demo](#live-demo) section.
2.  **Upload an Image:** Click on the "Browse files" button or drag and drop an image of a plant leaf into the uploader.
3.  **Get the Diagnosis:** The application will process the image and instantly display the predicted disease along with the model's confidence score.

---

### ⚙️ Installation & Local Setup

To run this project on your local machine, please follow these steps:

1.  **Clone the Repository:**
    ```bash
    git clone [https://github.com/Rpokhariya/Plant-Disease-Detection.git](https://github.com/Rpokhariya/Plant-Disease-Detection.git)
    cd Plant-Disease-Detection
    ```

2.  **Create a Virtual Environment (Recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3.  **Install Dependencies:**
    Make sure you have all the required libraries by installing them from the `requirements.txt` file.
    ```bash
    pip install -r requirements.txt
    ```

4.  **Run the Streamlit Application:**
    Execute the following command in your terminal:
    ```bash
    streamlit run app.py
    ```
    This will open the application in your default web browser.

---





