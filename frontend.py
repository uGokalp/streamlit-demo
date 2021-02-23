"""
This model implements a small web app that allows the user to
run predictions on MNIST, and cats and dogs like images.
"""
import streamlit as st
import imageio
from PIL import Image, ImageEnhance
import numpy as np
from tensorflow import keras
from tempfile import NamedTemporaryFile
import pandas as pd

# Utils

def from_array(image: np.ndarray) -> 'pil image':
    """Use it to return a pil image from an array
    Without the try except I found that I run into issues after normalizing the image
    """
    try:
        im = Image.fromarray(image)
    except TypeError:
        im = Image.fromarray((image * 255).astype(np.uint8))
    return im

def pad_to(image, size:int) -> 'pil-image':
    """
    Pads the image to specified size(square)
    If you have a 28x28 image and you call pad_to with size 128
    it will return a 128x128 image
    """
    current_size = image.size
    ratio = float(size)/max(current_size)
    new_size = tuple([int(x*ratio) for x in current_size])
    image = image.resize(new_size, Image.ANTIALIAS)
    new_im = Image.new("RGB", (size, size))
    new_im.paste(image, ((size-new_size[0])//2,
                (size-new_size[1])//2))
    return new_im

# Small helper for loading image
def load_image(buffer) -> np.ndarray: return np.array(Image.open(buffer))

# I don't like calling np.array on a image so here it is
def to_numpy(image: 'pil'):
    return np.array(image)

# Caches the output
@st.cache(allow_output_mutation=True)
def load_model(model_name:str):
    """
    Returns a simple keras model by reading it first.
    """
    if model_name == "Cats & Dogs":
        model = keras.models.load_model('xception_cat_dogs.h5')
    elif model_name ==  "MNIST":
        model = keras.models.load_model('mnist.h5')
    return model

def get_dataframe(outcome:dict) -> pd.DataFrame:
    """
    Returns a sorted dataframe.
    """
    df = pd.DataFrame(outcome.values(),
                index=pd.Series(outcome.keys(), name='Target'),
                columns=["Probabilities"])\
                .sort_values("Probabilities", ascending=False)
    return df


def predict(model_name, model, im) -> tuple:
    pred = model.predict(im)
    if model_name == "Cats & Dogs":
        outcome = {"dog":pred[0][0], "cat":1-pred[0][0]}
        return get_dataframe(outcome)
    elif model_name == "MNIST":
        outcome = {str(num):pred for num,pred in zip(range(len(pred[0])),pred[0])}
        return get_dataframe(outcome)
    

def reshape_image(im, model):
    shape = im.shape
    image = from_array(im)
    if model == "Cats & Dogs":
        if shape[0] > 28:
            image = to_numpy(image.resize((150,150)))
        else:
            image = to_numpy(pad_to(image, 150))
    elif model == "MNIST":
        try:
            image = to_numpy(image.resize((28,28)).convert('L')).reshape((28,28,1))
        except Exception as e:
            print(e)
    return image / 255

# Models require a batch layer
def add_batch(im:np.ndarray): return np.expand_dims(im, axis=0)

def activate_checkbox(checkbox):
    """
    Checks if any of the checkboxes are activated
    If so runs them.
    """
    global main_img
    global im_preview
    if any(checkbox_holder.values()):
        for augmentation, checkbox in checkbox_holder.items():
            if checkbox:
                im_preview = AUGMENTATION_TO_FUNCTION[augmentation](im_preview)
    main_img = main_img.image(im_preview, clamp=True)

# Augmentations

def horizontal_flip(image: np.ndarray):
    return np.fliplr(image)

def noise(image: np.ndarray):
    mean = 0.0   
    std = 0.1  
    noisy_img = image + np.random.normal(mean, std, image.shape)
    noisy_img_clipped = np.clip(noisy_img, 0, 255)
    return noisy_img_clipped
    
def rotation(image: np.ndarray):return np.rot90(image)

def brightness(image: np.ndarray):
    enchancer = ImageEnhance.Brightness(from_array(image))
    bright_im = enchancer.enhance(1.5)
    return np.array(bright_im)

# Constants
MODEL = {"Cats & Dogs": load_model("Cats & Dogs"), "MNIST": load_model("MNIST")}

MODEL_OPTIONS_MAP = {
    "Cats & Dogs": "xception_cat_dogs.h5",
    "MNIST": "mnist.ht",
}
MODEL_OPTIONS = ("Cats & Dogs", "MNIST")

AUGMENTATIONS = ("Flip", "Noise", "Rotation", "Brightness",)
AUGMENTATION_TO_FUNCTION = dict.fromkeys(AUGMENTATIONS)
AUGMENTATION_TO_FUNCTION["Flip"] = horizontal_flip
AUGMENTATION_TO_FUNCTION["Noise"] = noise
AUGMENTATION_TO_FUNCTION["Rotation"] = rotation
AUGMENTATION_TO_FUNCTION["Brightness"] = brightness

# Main
st.title("My fantastic app")

st.sidebar.title(f"Model selection")
current_model = st.sidebar.selectbox("Which model would you like to pick?", MODEL_OPTIONS)

st.sidebar.write(f"Current model is: {current_model}")
image_buffer = st.sidebar.file_uploader("Upload Image", type=["png", "jpg", "jpeg"])
main_img = st.empty()
prediction_text = st.empty()

st.sidebar.title(f"Image augmentations")
checkbox_holder = {augmentation: st.sidebar.checkbox(augmentation) for augmentation in AUGMENTATIONS}
prediction_text = st.empty()
apply_button = st.sidebar.button("Apply")
predict_button = st.sidebar.button("Predict")

if image_buffer:
    im_preview = load_image(image_buffer)
    main_img = main_img.image(im_preview)

if apply_button: activate_checkbox(checkbox_holder)

if predict_button:
    activate_checkbox(checkbox_holder)
    
    main_img = main_img.image(im_preview, clamp=True)
    model = MODEL.get(current_model)
    predict_image = reshape_image(im_preview, current_model)
    print("Current model", current_model)
    print("Incoming image, ",predict_image.shape)
    pred = predict(current_model, model, add_batch(predict_image))
    prediction_text.write(pred)
