import numpy as np
import cv2
import gradio as gr
from tensorflow.keras.utils import img_to_array
from tensorflow.keras.models import load_model
import os
# Load Model..
model = load_model(r"plant_leaf_diseases_model.h5")

classes = ['Pepper_bell_Bacterial_spot',
 'Pepper_bell_healthy',
 'Potato_Early_blight',
 'Potato_Late_blight',
 'Potato_healthy',
 'Tomato_Bacterial_spot',
 'Tomato_Early_blight',
 'Tomato_Late_blight',
 'Tomato_Leaf_Mold',
 'Tomato_Septoria_leaf_spot',
 'Tomato_Spider_mites_Two_spotted_spider_mite',
 'Tomato_Target_Spot',
 'Tomato_Tomato_YellowLeaf_Curl_Virus',
 'Tomato_Tomato_mosaic_virus',
 'Tomato_healthy']


def predict_image(img):
    
    x = img_to_array(img)
    x = cv2.resize(x, (256, 256), interpolation=cv2.INTER_AREA)
    x /= 255
    x = np.expand_dims(x, axis=0)
    image = np.vstack([x])
    prediction = model.predict(image)[0]  # Get the predictions for the first (and only) image
    class_probabilities = {class_name: prob for class_name, prob in zip(classes, prediction)}
    formatted_class_probabilities = {class_name: "{:.2f}".format(prob) for class_name, prob in class_probabilities.items()}
    return formatted_class_probabilities

# Define the Gradio Interface with the desired title and description

description_html = """
<p>Model trained for educational purposes only; usage subject to terms and conditions.</p>

"""

# Define example images and their true labels for users to choose from
example_data = [
    r"assets/examples/images/1.JPG",
    r"assets/examples/images/2.JPG",
    r"assets/examples/images/3.JPG",
    r"assets/examples/images/4.JPG",
    r"assets/examples/images/5.JPG",
    r"assets/examples/images/6.JPG",
    r"assets/examples/images/7.JPG",
    r"assets/examples/images/8.JPG",
    r"assets/examples/images/9.JPG",
    r"assets/examples/images/10.JPG"
]



gr.Interface(
    fn=predict_image,
    inputs="image",
    outputs=gr.Label(num_top_classes=15,min_width=360),
    title="<center><b>🪴Plant Diseases ©️nn🪴</b></center>",
    description=description_html,
    allow_flagging='never',
    examples=example_data 
).launch()
