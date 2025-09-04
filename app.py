import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
from huggingface_hub import hf_hub_download

st.title("Images Classification: Cats vs Dogs")

model_path = hf_hub_download("hilyashfae/catsvsdogs", filename="catsvsdogs.h5")

@st.cache_resource
def load_model(path):
    return tf.keras.models.load_model(path, compile=False)

model = load_model(model_path)

def preprocess(img):
    target_size = (model.input_shape[1], model.input_shape[2])
    img = img.resize(target_size)
    img = np.array(img) / 255.0
    return np.expand_dims(img, axis = 0)

if "pred" not in st.session_state:
    st.session_state.pred = None

with st.container(border=True):
    st.markdown("### Upload & Preview")
    uploaded = st.file_uploader("Upload Images", type=["jpg", "jpeg"], label_visibility="collapsed")

    preview_img = None
    if uploaded:
        preview_img = Image.open(uploaded).convert("RGB")
        st.image(preview_img, caption="Image is uploaded!", use_container_width=True)
    else:

        st.session_state.pred = None


with st.container(border=True):
    st.markdown("### Prediction Result")

    if preview_img is None:
        st.info("Please upload an image first to display the prediction results.")
    else:
        if st.button("Predict", type="primary"):
            x = preprocess(preview_img)
            pred = model.predict(x, verbose=0)

            if pred.shape[1] == 1:
                p_dog = float(pred[0][0])
                p_cat = 1.0 - p_dog
            else:
                p_dog = float(pred[0][0])
                p_cat = float(pred[0][1])

            label = "It's a dog" if p_dog >= p_cat else "It's a cat"
            st.session_state.pred = {
                "label": label,
                "p_dog": p_dog,
                "p_cat": p_cat
            }

    if st.session_state.pred is not None:
        st.subheader(st.session_state.pred["label"])
        c1, c2 = st.columns(2)
        with c1:
            st.write(f"**Dog Probability**: {st.session_state.pred['p_dog']*100:.2f}%")
            st.progress(int(round(st.session_state.pred['p_dog']*100)))
        with c2:
            st.write(f"**Cat Probability**: {st.session_state.pred['p_cat']*100:.2f}%")
            st.progress(int(round(st.session_state.pred['p_cat']*100)))