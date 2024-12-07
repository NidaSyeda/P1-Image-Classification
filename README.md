# Image Classification App with MobileNetV2 and CIFAR-10

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
import tensorflow as tf
from PIL import Image
import numpy as np
from model import load_model
from utils import preprocess_image

# Set page config
st.set_page_config(
    page_title="Image Classification App",
    page_icon="🔍",
    layout="wide"
)

# Title and description
st.title("Image Classification with MobileNetV2")
st.write("Upload an image to classify it into one of the CIFAR-10 classes")

# Load model
@st.cache_resource
def get_model():
    return load_model()

model = get_model()

# Class labels
class_labels = ['airplane', 'automobile', 'bird', 'cat', 'deer', 
                'dog', 'frog', 'horse', 'ship', 'truck']

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display and process image
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Uploaded Image")
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)
    
    # Preprocess image
    processed_image = preprocess_image(image)
    
    # Make prediction
    prediction = model.predict(processed_image)
    predicted_class = class_labels[np.argmax(prediction[0])]
    confidence = float(np.max(prediction[0]))
    
    with col2:
        st.subheader("Prediction Results")
        st.write(f"Predicted Class: **{predicted_class}**")
        st.write(f"Confidence: **{confidence:.2%}**")
        
        # Display confidence bars for all classes
        st.subheader("Confidence Scores")
        for i, (label, score) in enumerate(zip(class_labels, prediction[0])):
            st.progress(float(score))
            st.write(f"{label}: {float(score):.2%}")
```

## Contributing

1. Fork the repository
2. Create a new branch for your feature
3. Commit your changes
4. Push to the branch
5. Create a new Pull Request

## Screenshot


