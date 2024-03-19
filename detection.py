# importing libraries
import numpy as np
import cv2
from PIL import Image
import nibabel as nib
import tensorflow as tf
from tensorflow.keras.models import load_model


# loading model
model = load_model('effnet.h5')
    


# Function for JPEG file prediction and classification
def img_pred(image):
    if isinstance(image,tuple):
        pil_image=Image.open(image[0])
    else:
        pil_image = Image.open(image)
        image = np.array(pil_image)
    # image = np.array(image, dtype=np.uint8)
    opencvImage = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    img = cv2.resize(opencvImage,(150,150))
    img = np.expand_dims(img, axis=0)
    img = img.reshape(1,150,150,3)
    p = model.predict(img)
    p = np.argmax(p,axis=1)[0]
    if p==0:
        p='Glioma Tumor'
    elif p==1:
        return ('No tumor')
    elif p==2:
        p='Meningioma Tumor'
    else:
        p='Pituitary Tumor'

    if p!=1:
        print(f'The Model predicts that it is a {p}')
    
    return p


# Function for NIfTI file prediction and 3D visualization
def predict_nii(nii_path):
    # Load NIfTI file
    # img = 
    img_data = np.array(nib.load(nii_path).get_fdata())
    # prediction
    out=[]
    for i in range(0,155):
        img=img_data[:, :, i]
        img_converted = cv2.convertScaleAbs(img, alpha=255.0)
        opencvImage = cv2.cvtColor(np.array(img_converted), cv2.COLOR_RGB2BGR)
        img = cv2.resize(opencvImage,(150,150))
        img = np.expand_dims(img, axis=0)
        img = img.reshape(1,150,150,3)
        p = model.predict(img)
        p = np.argmax(p,axis=1)[0]
        if p==0:
            p='Glioma Tumor'
            out.append(p)
            break
        elif p==1:
            # predicitions.append('no tumor')
            # predicitions[f'slice{i}']='no tumor'
            pass
        elif p==2:
            p='Meningioma Tumor'
            out.append(p)
            break
        else:
            p='Pituitary Tumor'
            out.append(p)
            break
    if len(out)==0:
        out.append("no Tumour")
    tumor_type=out[0]
    return (tumor_type)


