import streamlit as st
from PIL import Image
from sentence_transformers import SentenceTransformer



uploaded_file = st.file_uploader("Upload image", type=[
                                     "png", "jpeg", "jpg"], 
                                     accept_multiple_files=False, 
                                     key=None, help="upload image")

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image.', use_column_width=True)
    st.write("")
    #Load CLIP model
    model = SentenceTransformer('clip-ViT-B-32')
    #Encode a text and image
    embedding = model.encode(image)
    st.table(embedding)

