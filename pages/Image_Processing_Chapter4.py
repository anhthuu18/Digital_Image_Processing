import sys
from PIL import Image
import cv2
import numpy as np
sys.path.append('../library')
from library.bosung_streamlit.sidebar import *
from library.bosung_streamlit.Chapter04 import *

# Sửa lại list ảnh cho đúng với yêu cầu
list_images_c4_0 = ['','1. Spectrum.png','2. Highpass filter.png','','3. Moire.png']

def XuLyAnh():
    global image_file
    global c1,c2
    st.title("Xử lý ảnh - "+Get_XyLyAnh_C4_0())
    image_file = st.file_uploader("Upload Images", type=["bmp", "png","jpg","jpeg"])
    c1,c2 = st.columns(2)

imgin = None
def InsertImage(image_file,path=''):
    global image
    if image_file is not None:
        image = Image.open(image_file)
        c1.image(image, caption=None)
    else:
        image = Image.open("./images/ImageProcessing_Chapter4/"+path)
        c1.image(image, caption=None)
        image.close()

def Chosen_Proccesing(option,path=''):
    global imgin
    if(option != '3. Vẽ bộ lọc note-reject'):  # Giữ nguyên điều kiện đặc biệt này
        if image_file is not None:
            # Mở lại file để xử lý
            image = Image.open(image_file)
            image_array = np.array(image)
            # Chuyển sang grayscale nếu là ảnh màu
            if len(image_array.shape) == 3:
                imgin = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
            else:
                imgin = image_array
            image.close()
        else:
            InsertImage(image_file,list_images_c4_0[Get_Index_Image_C4_0()])
            imgin = cv2.imread("./images/ImageProcessing_Chapter4/"+path, cv2.IMREAD_GRAYSCALE)

    if(option == '1. Spectrum'):
        Image_Spectrum()
    elif(option == '2. Lọc trong miền tần số'):
        Image_FrequencyFilter()
    elif(option == '3. Vẽ bộ lọc note-reject'):
        Image_CreateNotchRejectFilter()
    elif(option == '4. Xóa nhiễu Moire'):
        Image_RemoveMoire()

def Image_All():
    global image_file
    if(Get_XyLyAnh_C4_0()!=''):
        if(image_file is not None):
            # Mở để xử lý
            InsertImage(image_file)
            Chosen_Proccesing(Get_XyLyAnh_C4_0())  # Bỏ comment dòng này
        else:
            Chosen_Proccesing(Get_XyLyAnh_C4_0(),list_images_c4_0[Get_Index_Image_C4_0()])

def Image_Spectrum():
    output_image = Spectrum(imgin)
    c2.image(output_image, caption=None)

def Image_FrequencyFilter():
    output_image = FrequencyFilter(imgin)
    c2.image(output_image, caption=None)

def Image_CreateNotchRejectFilter():
    output_image = CreateNotchRejectFilter()
    c2.image(output_image, caption=None)

def Image_RemoveMoire():
    output_image = RemoveMoire(imgin)
    c2.image(output_image, caption=None)

if __name__=="__main__":
    configure()
    cs_sidebar()
    XuLyAnh()
    Image_All()