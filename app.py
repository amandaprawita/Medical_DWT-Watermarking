"""
Streamlit Cheat Sheet

App to summarise streamlit docs v1.25.0

There is also an accompanying png and pdf version

https://github.com/daniellewisDL/streamlit-cheat-sheet

v1.25.0
20 August 2023

Author:
    @daniellewisDL : https://github.com/daniellewisDL

Contributors:
    @arnaudmiribel : https://github.com/arnaudmiribel
    @akrolsmir : https://github.com/akrolsmir
    @nathancarter : https://github.com/nathancarter

"""

import streamlit as st
from pathlib import Path
import base64
# from Tabs import Watermarking, Extract
import streamlit as st
import numpy as np
from scipy.fftpack import dct, idct
import cv2 as cv
import random
import math
from PIL import Image
# Add this import at the beginning of your code
import streamlit as st
import qrcode
import hashlib
import streamlit as st
import cv2
import numpy as np
import pywt
from PIL import Image
import imagehash

# Initial page config

# from streamlit_option_menu import option_menu

# with st.sidebar:
#     selected = option_menu("Main Menu", ["Home", 'Settings'], 
#         icons=['house', 'gear'], menu_icon="cast", default_index=1)
#     selected

def extract_watermark(watermarked_image, original_image, strength=0.5):
    # Convert the images to grayscale
    watermarked_gray = cv2.cvtColor(watermarked_image, cv2.COLOR_BGR2GRAY)
    original_gray = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)

    # Apply DWT to the watermarked image
    coeffs_watermarked = pywt.dwt2(watermarked_gray, 'bior1.3')
    LL_watermarked, (LH_watermarked, HL_watermarked, HH_watermarked) = coeffs_watermarked

    # Apply DWT to the original image
    coeffs_original = pywt.dwt2(original_gray, 'bior1.3')
    LL_original, _ = coeffs_original

    # Resize the original LL sub-band to match the size of the watermarked LL sub-band
    LL_original_resized = cv2.resize(LL_original, (LL_watermarked.shape[1], LL_watermarked.shape[0]))

    # Calculate the difference to extract the watermark
    LL_watermark = (LL_watermarked - LL_original_resized) / strength

    # Reconstruct the watermark sub-band
    coeffs_watermark = (LL_watermark, (LH_watermarked, HL_watermarked, HH_watermarked))
    extracted_watermark = pywt.idwt2(coeffs_watermark, 'bior1.3')

    # Normalize the extracted watermark
    extracted_watermark = normalize_image(extracted_watermark)

    # Convert to uint8 before thresholding
    extracted_watermark_uint8 = (extracted_watermark * 255).astype(np.uint8)

    # Additional processing to enhance the extracted watermark
    _, extracted_mask = cv2.threshold(extracted_watermark_uint8, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    extracted_watermark = cv2.bitwise_and(extracted_watermark_uint8, extracted_mask)

    # Additional processing for clearer extraction
    extracted_watermark = cv2.fastNlMeansDenoising(extracted_watermark, None, h=10, templateWindowSize=7, searchWindowSize=21)
    kernel = np.ones((3, 3), np.uint8)
    extracted_watermark = cv2.morphologyEx(extracted_watermark, cv2.MORPH_CLOSE, kernel)

    return extracted_watermark

def embed_watermark(host_image, watermark_image, strength=0.25  ):
    # Convert the images to grayscale
    host_gray = cv2.cvtColor(host_image, cv2.COLOR_BGR2GRAY)
    watermark_gray = cv2.cvtColor(watermark_image, cv2.COLOR_BGR2GRAY)

    # Apply DWT to the host image
    coeffs_host = pywt.dwt2(host_gray, 'bior1.3')
    LL_host, (LH_host, HL_host, HH_host) = coeffs_host

    # Apply DWT to the watermark image
    coeffs_watermark = pywt.dwt2(watermark_gray, 'bior1.3')
    LL_watermark, _ = coeffs_watermark

    # Resize the watermark LL sub-band to match the size of the host LL sub-band
    LL_watermark_resized = cv2.resize(LL_watermark, (LL_host.shape[1], LL_host.shape[0]))

    # Embed the watermark in the LL sub-band with a reduced strength
    LL_watermarked = LL_host + strength * LL_watermark_resized

    # Reconstruct the watermarked image
    coeffs_watermarked = (LL_watermarked, (LH_host, HL_host, HH_host))
    watermarked_image = pywt.idwt2(coeffs_watermarked, 'bior1.3')

    return watermarked_image

def normalize_image(image):
    # Normalize the image pixel values to be in the range [0.0, 1.0]
    image_min = np.min(image)
    image_max = np.max(image)

    normalized_image = (image - image_min) / (image_max - image_min)

    return normalized_image

# Function to generate hash value for an image
def generate_hash(img_path):
    with open(img_path, 'rb') as f:
        image_data = f.read()
        hash_value = hashlib.sha256(image_data).hexdigest()
    return hash_value

st.set_page_config(
     page_title='Medical Image Crypto',
     layout="wide",
     initial_sidebar_state="expanded",
)

def main():
    cs_sidebar()
    # cs_body()

    return None

# Thanks to streamlitopedia for the following code snippet

def img_to_bytes(img_path):
    img_bytes = Path(img_path).read_bytes()
    encoded = base64.b64encode(img_bytes).decode()
    return encoded

# sidebar

def generate_qr_code(text):
    qr = qrcode.QRCode(
        version=1,
        error_correction=qrcode.constants.ERROR_CORRECT_L,
        box_size=10,
        border=4,
    )
    qr.add_data(text)
    qr.make(fit=True)
    img = qr.make_image(fill_color="black", back_color="white")
    qr_path = f"{text}_qrcode.png"
    img.save(qr_path)
    return qr_path

def cs_sidebar():

    st.sidebar.markdown('''<img src='data:image/png;base64,{}' class='img-fluid' width=150 height=150>'''.format(img_to_bytes("madical.png")), unsafe_allow_html=True)
    st.sidebar.header('Medical Image Crypto')
    
    option = st.sidebar.selectbox(
        'Select Page : ', 
        ('Watermarking', 'Extract')
    )
    
    # Check the selected option and render the corresponding content
    if option == 'Watermarking':
        # st.title("Safe Medical")
        st.markdown("# Watermarking 🎈")
        
        input_text = st.text_input("Enter text to generate QR code:")

        col1, col2, col3 = st.columns(3)

        qr_path = None
        hash_value = None
        wmed = None
        
        if col1.button('Generate QR-Code'):
            qr_path = generate_qr_code(input_text)
            col1.image(qr_path, caption=f"Generated QR Code for: {input_text}", width=300)
            # col1.markdown(
            #     f'<div style="text-align: center;"><img src="{qr_path}" alt="Generated QR Code" width="250"><p>Generated QR Code for: {input_text}</p></div>',
            #     unsafe_allow_html=True
            # )
            
        # Upload the host image
        

        
        # Add a condition to check the dimensions of the uploaded image
        host_image = col2.file_uploader("Upload host image", type=["jpg", "jpeg", "png"])
        if host_image:
            img = Image.open(host_image)
            
            # Get the dimensions of the image
            img_width, img_height = img.size
            
            # Set the maximum allowed dimensions
            max_width, max_height = 1000, 1000
            
            if img_width > max_width or img_height > max_height:
                st.warning(f"Warning: The uploaded image dimensions ({img_width}x{img_height}) exceed the recommended maximum size of {max_width}x{max_height} pixels.")
            
        watermark_image = qr_path
        
        if host_image:
            col2.image(host_image, caption="Uploaded Image", width=300)
        
        col3.subheader('Watermarked Image')
        
        if host_image and watermark_image:
            # Convert uploaded images to OpenCV format
            # host_image_cv = cv2.imdecode(np.frombuffer(host_image.read(), np.uint8), 1)
            # watermark_image_cv = cv2.imdecode(np.frombuffer(watermark_image.read(), np.uint8), 1)

            host_image_cv = cv2.imdecode(np.frombuffer(host_image.read(), np.uint8), 1)
            watermark_image_cv = cv2.imread(watermark_image)
            
            # Display the uploaded images
            # col2.image(host_image_cv, caption="Host Image", use_column_width=True)
            # st.image(watermark_image_cv, caption="Watermark Image", use_column_width=True)

            # Embed watermark with reduced strength
            watermarked_image_cv = embed_watermark(host_image_cv, watermark_image_cv, strength=0.05)

            # Normalize the watermarked image
            watermarked_image_normalized = normalize_image(watermarked_image_cv)

            col3.image(watermarked_image_normalized, caption="Watermarked Image", use_column_width=True)

            # Save the watermarked image
            # Save the watermarked image
            cv2.imwrite("watermarked_image.jpg", (watermarked_image_normalized * 255).astype(np.uint8))
                
            # # Display Watermarked Image
            # # Display Watermarked Image
            # col3.image(np.clip(wmed.astype(float) / 255.0, 0.0, 1.0), caption="Watermarked Image", width=300)
            
            # # Save the watermarked image
            # cv.imwrite(watermarked_img, wmed)
            
            # Generate hash value for the watermarked image
            # Generate hash value for the watermarked image
            
            hash_value = imagehash.average_hash(Image.open('watermarked_image.jpg'))
            # img = "watermarked_image.jpg"
            # hash_value = generate_hash(img)

        # Optimize performance
        
        # Save the watermarked image
        
        col3.subheader('Hash Values')
        # col3.write('Cache data objects')
        
        col3.success(hash_value)
        
        # Add your Watermarking content here
    elif option == 'Extract':
        st.markdown("# Extract 🩺")
        st.subheader('Information')
        
        col1, col2 = st.columns(2)
        
        # ...
        
        
        #uploaded_img = col2.file_uploader("Upload image")
        uploaded_img_water = col1.file_uploader("Upload watermark image", type=["jpg", "jpeg", "png"])
        original_image = col1.file_uploader("Upload original host image", type=["jpg", "jpeg", "png"])
        #uploaded_img = original_image
        # if uploaded_img_water:
        hash_value = None
        # else:
        #     st.warning("Please upload an image for watermark extraction.")
        #if uploaded_img:
            #hash_value = imagehash.average_hash(Image.open(uploaded_img))
        
        if uploaded_img_water and original_image:
            hash_value = imagehash.average_hash(Image.open(original_image))

            watermarked_image_cv = cv2.imdecode(np.frombuffer(uploaded_img_water.read(), np.uint8), 1)
            original_image_cv = cv2.imdecode(np.frombuffer(original_image.getvalue(), np.uint8), 1)
            original_image_cv = cv2.resize(original_image_cv, (watermarked_image_cv.shape[1], watermarked_image_cv.shape[0]))

            # Convert BGR to RGB for displaying with Streamlit
            watermarked_image_cv_rgb = cv2.cvtColor(watermarked_image_cv, cv2.COLOR_BGR2RGB)
            original_image_cv_rgb = cv2.cvtColor(original_image_cv, cv2.COLOR_BGR2RGB)

            # st.image(watermarked_image_cv_rgb, caption="Watermarked Image", use_column_width=True)
            # st.image(original_image_cv_rgb, caption="Original Image", use_column_width=True)

            extracted_watermark_cv = extract_watermark(watermarked_image_cv, original_image_cv, strength=0.1)

            # Invert the extracted watermark
            extracted_watermark_cv = cv2.bitwise_not(extracted_watermark_cv)

            # Convert to RGB for displaying with Streamlit
            extracted_watermark_cv_rgb = cv2.cvtColor(extracted_watermark_cv, cv2.COLOR_BGR2RGB)

            # Display the inverted extracted watermark
            col1.image(extracted_watermark_cv_rgb, caption="Extracted Watermark", use_column_width=True, channels="GRAY")

            col1.text("Scan the QR-Code for Information Details")
            
        col2.subheader('Image Verification')
        hash_input = col2.text_input('Input hash image value: ')

        # Compare the hash values
        if str(hash_value) == hash_input:
            col2.success("Hash values match. Image integrity is verified.")
        else:
            col2.error("Hash values do not match. Image integrity may be compromised.")
                
        
        
        
        
        # if uploaded_img_water and original_image:
        #     watermarked_image_cv = cv2.imdecode(np.frombuffer(uploaded_img_water.read(), np.uint8), 1)
        #     original_image_cv = cv2.imdecode(np.frombuffer(original_image.read(), np.uint8), 1)
        #     original_image_cv = cv2.resize(original_image_cv, (watermarked_image_cv.shape[1], watermarked_image_cv.shape[0]))

        #     # Convert BGR to RGB for displaying with Streamlit
        #     watermarked_image_cv_rgb = cv2.cvtColor(watermarked_image_cv, cv2.COLOR_BGR2RGB)
        #     original_image_cv_rgb = cv2.cvtColor(original_image_cv, cv2.COLOR_BGR2RGB)

        #     # st.image(watermarked_image_cv_rgb, caption="Watermarked Image", use_column_width=True)
        #     # st.image(original_image_cv_rgb, caption="Original Image", use_column_width=True)

        #     extracted_watermark_cv = extract_watermark(watermarked_image_cv, original_image_cv, strength=0.1)

        #     # Invert the extracted watermark
        #     extracted_watermark_cv = cv2.bitwise_not(extracted_watermark_cv)

        #     # Convert to RGB for displaying with Streamlit
        #     extracted_watermark_cv_rgb = cv2.cvtColor(extracted_watermark_cv, cv2.COLOR_BGR2RGB)

        #     # Display the inverted extracted watermark
        #     col1.image(extracted_watermark_cv_rgb, caption="Extracted Watermark", use_column_width=True, channels="GRAY")

        #     col1.text("Scan the QR-Code for Information Details")
            
        
        
    return None

if __name__ == '__main__':
    main()


# """
# Streamlit Cheat Sheet

# App to summarise streamlit docs v1.25.0

# There is also an accompanying png and pdf version

# https://github.com/daniellewisDL/streamlit-cheat-sheet

# v1.25.0
# 20 August 2023

# Author:
#     @daniellewisDL : https://github.com/daniellewisDL

# Contributors:
#     @arnaudmiribel : https://github.com/arnaudmiribel
#     @akrolsmir : https://github.com/akrolsmir
#     @nathancarter : https://github.com/nathancarter

# """

# import streamlit as st
# from pathlib import Path
# import base64
# # from Tabs import Watermarking, Extract
# import streamlit as st
# import numpy as np
# from scipy.fftpack import dct, idct
# import cv2 as cv
# import random
# import math
# from PIL import Image
# # Add this import at the beginning of your code
# import streamlit as st
# import qrcode
# import hashlib
# import streamlit as st
# import cv2
# import numpy as np
# import pywt
# from PIL import Image
# import imagehash

# # Initial page config

# # from streamlit_option_menu import option_menu

# # with st.sidebar:
# #     selected = option_menu("Main Menu", ["Home", 'Settings'], 
# #         icons=['house', 'gear'], menu_icon="cast", default_index=1)
# #     selected

# def extract_watermark(watermarked_image, original_image, strength=0.5):
#     # Convert the images to grayscale
#     watermarked_gray = cv2.cvtColor(watermarked_image, cv2.COLOR_BGR2GRAY)
#     original_gray = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)

#     # Apply DWT to the watermarked image
#     coeffs_watermarked = pywt.dwt2(watermarked_gray, 'bior1.3')
#     LL_watermarked, (LH_watermarked, HL_watermarked, HH_watermarked) = coeffs_watermarked

#     # Apply DWT to the original image
#     coeffs_original = pywt.dwt2(original_gray, 'bior1.3')
#     LL_original, _ = coeffs_original

#     # Resize the original LL sub-band to match the size of the watermarked LL sub-band
#     LL_original_resized = cv2.resize(LL_original, (LL_watermarked.shape[1], LL_watermarked.shape[0]))

#     # Calculate the difference to extract the watermark
#     LL_watermark = (LL_watermarked - LL_original_resized) / strength

#     # Reconstruct the watermark sub-band
#     coeffs_watermark = (LL_watermark, (LH_watermarked, HL_watermarked, HH_watermarked))
#     extracted_watermark = pywt.idwt2(coeffs_watermark, 'bior1.3')

#     # Normalize the extracted watermark
#     extracted_watermark = normalize_image(extracted_watermark)

#     # Convert to uint8 before thresholding
#     extracted_watermark_uint8 = (extracted_watermark * 255).astype(np.uint8)

#     # Additional processing to enhance the extracted watermark
#     _, extracted_mask = cv2.threshold(extracted_watermark_uint8, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
#     extracted_watermark = cv2.bitwise_and(extracted_watermark_uint8, extracted_mask)

#     # Additional processing for clearer extraction
#     extracted_watermark = cv2.fastNlMeansDenoising(extracted_watermark, None, h=10, templateWindowSize=7, searchWindowSize=21)
#     kernel = np.ones((3, 3), np.uint8)
#     extracted_watermark = cv2.morphologyEx(extracted_watermark, cv2.MORPH_CLOSE, kernel)

#     return extracted_watermark

# def embed_watermark(host_image, watermark_image, strength=0.5):
#     # Convert the images to grayscale
#     host_gray = cv2.cvtColor(host_image, cv2.COLOR_BGR2GRAY)
#     watermark_gray = cv2.cvtColor(watermark_image, cv2.COLOR_BGR2GRAY)

#     # Apply DWT to the host image
#     coeffs_host = pywt.dwt2(host_gray, 'bior1.3')
#     LL_host, (LH_host, HL_host, HH_host) = coeffs_host

#     # Apply DWT to the watermark image
#     coeffs_watermark = pywt.dwt2(watermark_gray, 'bior1.3')
#     LL_watermark, _ = coeffs_watermark

#     # Resize the watermark LL sub-band to match the size of the host LL sub-band
#     LL_watermark_resized = cv2.resize(LL_watermark, (LL_host.shape[1], LL_host.shape[0]))

#     # Embed the watermark in the LL sub-band with a reduced strength
#     LL_watermarked = LL_host + strength * LL_watermark_resized

#     # Reconstruct the watermarked image
#     coeffs_watermarked = (LL_watermarked, (LH_host, HL_host, HH_host))
#     watermarked_image = pywt.idwt2(coeffs_watermarked, 'bior1.3')

#     return watermarked_image

# def normalize_image(image):
#     # Normalize the image pixel values to be in the range [0.0, 1.0]
#     image_min = np.min(image)
#     image_max = np.max(image)

#     normalized_image = (image - image_min) / (image_max - image_min)

#     return normalized_image

# # Function to generate hash value for an image
# def generate_hash(img_path):
#     with open(img_path, 'rb') as f:
#         image_data = f.read()
#         hash_value = hashlib.sha256(image_data).hexdigest()
#     return hash_value

# st.set_page_config(
#      page_title='Medical Image Crypto',
#      layout="wide",
#      initial_sidebar_state="expanded",
# )

# def main():
#     cs_sidebar()
#     # cs_body()

#     return None

# # Thanks to streamlitopedia for the following code snippet

# def img_to_bytes(img_path):
#     img_bytes = Path(img_path).read_bytes()
#     encoded = base64.b64encode(img_bytes).decode()
#     return encoded

# # sidebar

# def generate_qr_code(text):
#     qr = qrcode.QRCode(
#         version=1,
#         error_correction=qrcode.constants.ERROR_CORRECT_L,
#         box_size=10,
#         border=4,
#     )
#     qr.add_data(text)
#     qr.make(fit=True)
#     img = qr.make_image(fill_color="black", back_color="white")
#     qr_path = f"{text}_qrcode.png"
#     img.save(qr_path)
#     return qr_path

# def cs_sidebar():

#     st.sidebar.markdown('''<img src='data:image/png;base64,{}' class='img-fluid' width=150 height=150>'''.format(img_to_bytes("madical.png")), unsafe_allow_html=True)
#     st.sidebar.header('Medical Image Crypto')
    
#     option = st.sidebar.selectbox(
#         'Select Page : ', 
#         ('Watermarking', 'Extract')
#     )
    
#     # Check the selected option and render the corresponding content
#     if option == 'Watermarking':
#         # st.title("Safe Medical")
#         st.markdown("# Watermarking 🎈")
        
#         input_text = st.text_input("Enter text to generate QR code:")

#         col1, col2, col3 = st.columns(3)

#         qr_path = None
#         hash_value = None
#         wmed = None
        
#         if col1.button('Generate QR-Code'):
#             qr_path = generate_qr_code(input_text)
#             col1.image(qr_path, caption=f"Generated QR Code for: {input_text}", width=300)
#             # col1.markdown(
#             #     f'<div style="text-align: center;"><img src="{qr_path}" alt="Generated QR Code" width="250"><p>Generated QR Code for: {input_text}</p></div>',
#             #     unsafe_allow_html=True
#             # )
            
#         # Upload the host image
        

        
#         # Add a condition to check the dimensions of the uploaded image
#         host_image = col2.file_uploader("Upload host image", type=["jpg", "jpeg", "png"])
#         if host_image:
#             img = Image.open(host_image)
            
#             # Get the dimensions of the image
#             img_width, img_height = img.size
            
#             # Set the maximum allowed dimensions
#             max_width, max_height = 1000, 1000
            
#             if img_width > max_width or img_height > max_height:
#                 st.warning(f"Warning: The uploaded image dimensions ({img_width}x{img_height}) exceed the recommended maximum size of {max_width}x{max_height} pixels.")
            
#         watermark_image = qr_path
        
#         if host_image:
#             col2.image(host_image, caption="Uploaded Image", width=300)
        
#         col3.subheader('Watermarked Image')
        
#         if host_image and watermark_image:
#             # Convert uploaded images to OpenCV format
#             # host_image_cv = cv2.imdecode(np.frombuffer(host_image.read(), np.uint8), 1)
#             # watermark_image_cv = cv2.imdecode(np.frombuffer(watermark_image.read(), np.uint8), 1)

#             host_image_cv = cv2.imdecode(np.frombuffer(host_image.read(), np.uint8), 1)
#             watermark_image_cv = cv2.imread(watermark_image)
            
#             # Display the uploaded images
#             # col2.image(host_image_cv, caption="Host Image", use_column_width=True)
#             # st.image(watermark_image_cv, caption="Watermark Image", use_column_width=True)

#             # Embed watermark with reduced strength
#             watermarked_image_cv = embed_watermark(host_image_cv, watermark_image_cv, strength=0.05)

#             # Normalize the watermarked image
#             watermarked_image_normalized = normalize_image(watermarked_image_cv)

#             col3.image(watermarked_image_normalized, caption="Watermarked Image", use_column_width=True)

#             # Save the watermarked image
#             # Save the watermarked image
#             cv2.imwrite("watermarked_image.jpg", (watermarked_image_normalized * 255).astype(np.uint8))
                
#             # # Display Watermarked Image
#             # # Display Watermarked Image
#             # col3.image(np.clip(wmed.astype(float) / 255.0, 0.0, 1.0), caption="Watermarked Image", width=300)
            
#             # # Save the watermarked image
#             # cv.imwrite(watermarked_img, wmed)
            
#             # Generate hash value for the watermarked image
#             # Generate hash value for the watermarked image
            
#             hash_value = imagehash.average_hash(Image.open('watermarked_image.jpg'))
#             # img = "watermarked_image.jpg"
#             # hash_value = generate_hash(img)

#         # Optimize performance
        
#         # Save the watermarked image
        
#         col3.subheader('Hash Values')
#         # col3.write('Cache data objects')
        
#         col3.success(hash_value)
        
#         # Add your Watermarking content here
#     elif option == 'Extract':
#         st.markdown("# Extract 🩺")
#         st.subheader('Information')
        
#         col1, col2 = st.columns(2)
        
#         # ...
        
        
#         uploaded_img = col2.file_uploader("Upload image")
#         uploaded_img_water = col1.file_uploader("Upload watermark image", type=["jpg", "jpeg", "png"])
#         original_image = col1.file_uploader("Upload original host image", type=["jpg", "jpeg", "png"])
#         # if uploaded_img_water:
#         hash_value = None
#         # else:
#         #     st.warning("Please upload an image for watermark extraction.")
#         if uploaded_img:
#             hash_value = imagehash.average_hash(Image.open(uploaded_img))
        
#         if uploaded_img_water and original_image:
#             watermarked_image_cv = cv2.imdecode(np.frombuffer(uploaded_img_water.read(), np.uint8), 1)
#             original_image_cv = cv2.imdecode(np.frombuffer(original_image.read(), np.uint8), 1)
#             original_image_cv = cv2.resize(original_image_cv, (watermarked_image_cv.shape[1], watermarked_image_cv.shape[0]))

#             # Convert BGR to RGB for displaying with Streamlit
#             watermarked_image_cv_rgb = cv2.cvtColor(watermarked_image_cv, cv2.COLOR_BGR2RGB)
#             original_image_cv_rgb = cv2.cvtColor(original_image_cv, cv2.COLOR_BGR2RGB)

#             # st.image(watermarked_image_cv_rgb, caption="Watermarked Image", use_column_width=True)
#             # st.image(original_image_cv_rgb, caption="Original Image", use_column_width=True)

#             extracted_watermark_cv = extract_watermark(watermarked_image_cv, original_image_cv, strength=0.1)

#             # Invert the extracted watermark
#             extracted_watermark_cv = cv2.bitwise_not(extracted_watermark_cv)

#             # Convert to RGB for displaying with Streamlit
#             extracted_watermark_cv_rgb = cv2.cvtColor(extracted_watermark_cv, cv2.COLOR_BGR2RGB)

#             # Display the inverted extracted watermark
#             col1.image(extracted_watermark_cv_rgb, caption="Extracted Watermark", use_column_width=True, channels="GRAY")

#             col1.text("Scan the QR-Code for Information Details")
            
#         col2.subheader('Image Verification')
#         hash_input = col2.text_input('Input hash image value: ')
        
#         # Compare the hash values
#         if str(hash_value) == hash_input:
#             col2.success("Hash values match. Image integrity is verified.")
#         else:
#             col2.error("Hash values do not match. Image integrity may be compromised.")
                
        
        
        
        
#         # if uploaded_img_water and original_image:
#         #     watermarked_image_cv = cv2.imdecode(np.frombuffer(uploaded_img_water.read(), np.uint8), 1)
#         #     original_image_cv = cv2.imdecode(np.frombuffer(original_image.read(), np.uint8), 1)
#         #     original_image_cv = cv2.resize(original_image_cv, (watermarked_image_cv.shape[1], watermarked_image_cv.shape[0]))

#         #     # Convert BGR to RGB for displaying with Streamlit
#         #     watermarked_image_cv_rgb = cv2.cvtColor(watermarked_image_cv, cv2.COLOR_BGR2RGB)
#         #     original_image_cv_rgb = cv2.cvtColor(original_image_cv, cv2.COLOR_BGR2RGB)

#         #     # st.image(watermarked_image_cv_rgb, caption="Watermarked Image", use_column_width=True)
#         #     # st.image(original_image_cv_rgb, caption="Original Image", use_column_width=True)

#         #     extracted_watermark_cv = extract_watermark(watermarked_image_cv, original_image_cv, strength=0.1)

#         #     # Invert the extracted watermark
#         #     extracted_watermark_cv = cv2.bitwise_not(extracted_watermark_cv)

#         #     # Convert to RGB for displaying with Streamlit
#         #     extracted_watermark_cv_rgb = cv2.cvtColor(extracted_watermark_cv, cv2.COLOR_BGR2RGB)

#         #     # Display the inverted extracted watermark
#         #     col1.image(extracted_watermark_cv_rgb, caption="Extracted Watermark", use_column_width=True, channels="GRAY")

#         #     col1.text("Scan the QR-Code for Information Details")
            
        
        
#     return None

# if __name__ == '__main__':
#     main()
