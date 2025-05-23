import sys
from PIL import Image
import cv2
import numpy as np
sys.path.append('../library')
from library.sidebar import *
from library.Chapter09 import *

list_images_c9 = ['','1. PhiLeGa.png','2. HatGao.png']

def XuLyAnh():
    global image_file
    global c1,c2
    st.title("Xử lý ảnh - "+Get_XyLyAnh_C9())
    image_file = st.file_uploader("Upload Images", type=["bmp", "png","jpg","jpeg"])
    c1,c2 = st.columns(2)

imgin = None
def InsertImage(image_file,path=''):
    global image
    if image_file is not None:
        # Không đóng file ở đây nữa
        image = Image.open(image_file)
        c1.image(image, caption=None)
    else:
        image = Image.open("./images/ImageProcessing_Chapter9/"+path)
        c1.image(image, caption=None)
        image.close()  # Chỉ đóng file với ảnh mặc định

def Chosen_Proccesing(option,path=''):
    global imgin
    if image_file is not None:
        # Mở lại file để xử lý
        image = Image.open(image_file)
        image_array = np.array(image)
        # Chuyển sang grayscale nếu là ảnh màu
        if len(image_array.shape) == 3:
            imgin = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
        else:
            imgin = image_array
        image.close()  # Đóng file sau khi xử lý xong
    else:
        InsertImage(image_file,list_images_c9[Get_Index_Image_C9()])
        imgin = cv2.imread("./images/ImageProcessing_Chapter9/"+path, cv2.IMREAD_GRAYSCALE)

    if option == '1. PhiLeGa':
        Image_ConnectedComponent()
    elif option == '2. HatGao':
        Image_Count_Right()

def Image_All():
    global image_file
    if Get_XyLyAnh_C9() != '':
        if image_file is not None:
            # Mở để xử lý
            InsertImage(image_file)
            Chosen_Proccesing(Get_XyLyAnh_C9())
        else:
            Chosen_Proccesing(Get_XyLyAnh_C9(),list_images_c9[Get_Index_Image_C9()])

def Image_Count_Right():
    text,output_image = CountRice(imgin)
    c2.image(output_image, caption=None)
    c2.markdown(text)

def Image_ConnectedComponent():
    text,output_image = ConnectedComponent(imgin)
    c2.image(output_image, caption=None)
    c2.markdown(text)

if __name__=="__main__":
    configure()
    cs_sidebar()
    XuLyAnh()
    Image_All()