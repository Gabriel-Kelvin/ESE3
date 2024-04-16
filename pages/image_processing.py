import streamlit as st
from PIL import Image
import numpy as np

def process_image(image, options):
    if "Resize" in options:
        image = image.resize((200, 200))
    if "Grayscale conversion" in options:
        image = image.convert("L")
    if "Image cropping" in options:
        image = image.crop((50, 50, 150, 150))
    if "Image rotation" in options:
        image = image.rotate(45)
    return image

# Main Streamlit application
st.title("Image Processing App")

# Upload image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    img = Image.open(uploaded_file)

    st.subheader("Original Image")
    st.image(img, caption="Original Image", use_column_width=True)

    st.subheader("Select Image Processing Techniques")
    options = st.multiselect("Select techniques:",
                             ["Resize", "Grayscale conversion", "Image cropping", "Image rotation"])


    if st.button("Process Image"):
        processed_img = process_image(img, options)
        st.subheader("Processed Image")
        st.image(processed_img, caption="Processed Image", use_column_width=True)










