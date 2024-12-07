# P1 Image Classification App with MobileNetV2 and CIFAR-10 (AICTE Internship 2024-25)

A Streamlit web application for image classification using MobileNetV2 pre-trained on CIFAR-10 dataset. This application allows users to upload images and get real-time predictions for 10 different classes.

## Features

- Real-time image classification using MobileNetV2
- Support for multiple image formats (PNG, JPG, JPEG)
- Pre-trained on CIFAR-10 dataset (10 classes)
- Interactive web interface built with Streamlit
- Prediction confidence scores visualization

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/image-classification-app.git
cd image-classification-app
```

2. Create a virtual environment and activate it:
```bash
python -m venv venv
source venv/bin/activate  # On Windows, use: venv\Scripts\activate
```

3. Install the required dependencies:
```bash
pip install -r requirements.txt
```

## Requirements

Create a `requirements.txt` file with the following dependencies:
```
streamlit==1.24.0
tensorflow==2.13.0
numpy==1.24.3
Pillow==9.5.0
```

## Project Structure

```
image-classification-app/
├── app.py
├── model.py
├── utils.py
├── requirements.txt
└── README.md
```

## Usage

1. Start the Streamlit app:
```bash
streamlit run app.py
```

2. Open your web browser and navigate to `http://localhost:8501`
3. Upload an image using the file uploader
4. View the prediction results and confidence scores

## Model Details

- Architecture: MobileNetV2
- Input Size: 32x32x3
- Number of Classes: 10
- Dataset: CIFAR-10
- Classes: ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

## Implementation Example

Here's a basic implementation of the main app (`app.py`):

```python
import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image

# Function for MobileNetV2 ImageNet model
def mobilenetv2_imagenet():
    st.title("Image Classification with MobileNetV2")
    
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png"])
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image', use_column_width=True)
        
        st.write("Classifying...")
        
        # Load MobileNetV2 model
        model = tf.keras.applications.MobileNetV2(weights='imagenet')
        
        # Preprocess the image
        img = image.resize((224, 224))
        img_array = np.array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = tf.keras.applications.mobilenet_v2.preprocess_input(img_array)
        
        # Make predictions
        predictions = model.predict(img_array)
        decoded_predictions = tf.keras.applications.mobilenet_v2.decode_predictions(predictions, top=1)[0]
        
        for i, (imagenet_id, label, score) in enumerate(decoded_predictions):
            st.write(f"{label}: {score * 100:.2f}%")

# Function for CIFAR-10 model
def cifar10_classification():
    st.title("CIFAR-10 Image Classification")
    
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png"])
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image', use_column_width=True)
        
        st.write("Classifying...")
        
        # Load CIFAR-10 model
        model = tf.keras.models.load_model('cifar10_model.h5')
        
        # CIFAR-10 class names
        class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
        
        # Preprocess the image
        img = image.resize((32, 32))
        img_array = np.array(img)
        img_array = img_array.astype('float32') / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        
        # Make predictions
        predictions = model.predict(img_array)
        predicted_class = np.argmax(predictions, axis=1)[0]
        confidence = np.max(predictions)
        
        st.write(f"Predicted Class: {class_names[predicted_class]}")
        st.write(f"Confidence: {confidence * 100:.2f}%")

# Main function to control the navigation
def main():
    st.sidebar.title("Navigation")
    choice = st.sidebar.selectbox("Choose Model", ("CIFAR-10","MobileNetV2 (ImageNet)"))
    
    if choice == "MobileNetV2 (ImageNet)":
        mobilenetv2_imagenet()
    elif choice == "CIFAR-10":
        cifar10_classification()

if __name__ == "__main__":
    main()
```

## Contributing

1. Fork the repository
2. Create a new branch for your feature
3. Commit your changes
4. Push to the branch
5. Create a new Pull Request

## Screenshots

### 1. Home Page
![Home Page](images/home_page.png)

### 2. Upload Image
![Upload Image](images/upload_image.png)

### 3. Prediction Results
![Prediction Results](images/prediction_results.png)


