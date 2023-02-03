import io
import requests
from PIL import Image
import streamlit as st

st.set_page_config(layout="wide")

def main():
    st.title("Pothole Classification Model")
    work = st.sidebar.selectbox("작업 선택", ("Image Classification", "Object Detection"))
    st.header(work)
    
    # uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

    # if uploaded_file:
    #     image_bytes = uploaded_file.getvalue()
    #     image = Image.open(io.BytesIO(image_bytes))

    #     st.image(image, caption='Uploaded Image')
    #     st.write("Classifying...")

    #     files = [
    #         ('files', (uploaded_file.name, image_bytes,
    #                    uploaded_file.type))
    #     ]
    loc = {"lat" :"122.21", "lng":"23.001"}
    response = requests.post("http://localhost:30001/predict", data=loc)
    # label = response.json()["result"]
    # st.write(f'label is {label}')
    print(response)
main()