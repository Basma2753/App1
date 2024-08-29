import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

CONFIGURATION = {
    "CLASS_NAMES": ['CaS', 'CoS', 'Gum', 'MC', 'OC', 'OLP', 'OT']
}

# Load the model
model = tf.keras.models.load_model("C:/Users/tarik/Downloads/model.keras")

# Streamlit app
st.title('Teeth Disease Classification App')

# Upload images
uploaded_files = st.file_uploader("Choose at least 5 images...", type="jpg", accept_multiple_files=True)

def preprocess_image(image):
    # Resize the image to 256x256 pixels
    image = cv2.resize(image, (256, 256))
    # Normalize the image
    image = np.expand_dims(image, axis=0)
    return image

if uploaded_files and len(uploaded_files) >= 5:
    st.write(f"Uploaded {len(uploaded_files)} images.")
    
    images = []
    for uploaded_file in uploaded_files:
        # Preprocess the uploaded image
        image = np.array(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(image, 1)
        images.append(image)
        st.image(image, channels="BGR")
    
    # Display images with true and predicted labels
    st.write("### True and Predicted Labels for Uploaded Images")
    plt.figure(figsize=(12, 12))

    for i, image in enumerate(images):
        ax = plt.subplot(4, 4, i + 1)
        preprocessed_image = preprocess_image(image)
        predictions = model.predict(preprocessed_image)
        predicted_label_index = int(tf.argmax(predictions, axis=-1).numpy()[0])
        predicted_label_name = CONFIGURATION["CLASS_NAMES"][predicted_label_index]
        
        plt.imshow(image / 255.0)
        plt.title(f"Predicted Label: {predicted_label_name}")
        plt.axis("off")
    
    st.pyplot(plt)
else:
    st.write("Please upload at least 5 images.")
