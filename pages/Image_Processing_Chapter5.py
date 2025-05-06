import sys
from PIL import Image
import cv2
import numpy as np
sys.path.append('../library')
from library.bosung_streamlit.sidebar import *
from library.bosung_streamlit.Chapter05 import *

# Danh sách ảnh mẫu cho Chapter 5
list_images_c5 = ['',
                  '1. TaoNhieuAnh.png',
                  '2. GoNhieuAnhItNhieu.png',
                  '3. GoNhieuAnhNhieuNhieu.png']

def XuLyAnh():
    """Khởi tạo giao diện chính"""
    global image_file
    global c1,c2
    st.title("Xử lý ảnh - "+Get_XyLyAnh_C5())
    image_file = st.file_uploader("Upload Images", type=["bmp", "png","jpg","jpeg"])
    c1,c2 = st.columns(2)

imgin = None
def InsertImage(image_file, path=''):
    """Hiển thị ảnh vào cột 1"""
    global image
    if image_file is not None:
        # Xử lý ảnh upload
        image = Image.open(image_file)
        c1.image(image, caption='Ảnh đã tải lên')
    else:
        # Xử lý ảnh mặc định
        image = Image.open("./images/ImageProcessing_Chapter5/"+path)
        c1.image(image, caption='Ảnh mặc định')
        image.close()

def Chosen_Proccesing(option, path=''):
    """Xử lý ảnh theo option được chọn"""
    global imgin
    if image_file is not None:
        # Xử lý ảnh upload
        image = Image.open(image_file)
        image_array = np.array(image)
        
        # Chuyển sang grayscale nếu là ảnh màu
        if len(image_array.shape) == 3:
            imgin = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
        else:
            imgin = image_array
        image.close()
    else:
        # Xử lý ảnh mặc định
        InsertImage(image_file, list_images_c5[Get_Index_Image_C5()])
        imgin = cv2.imread("./images/ImageProcessing_Chapter5/"+path, cv2.IMREAD_GRAYSCALE)

    # Chọn phương thức xử lý
    try:
        if option == '1. Tạo Nhiễu Ảnh':
            Image_CreateMotionNoise()
        elif option == '2. Lọc Ảnh Ít Nhiễu':
            Image_DenoiseMotion()
        elif option == '3. Lọc Ảnh Nhiều Nhiễu':
            Image_MoreDenoiseMotion()
    except Exception as e:
        st.error(f"Có lỗi xảy ra khi xử lý ảnh: {str(e)}")

def Image_All():
    """Hàm xử lý chính"""
    global image_file
    if Get_XyLyAnh_C5() != '':
        if image_file is not None:
            InsertImage(image_file)
            Chosen_Proccesing(Get_XyLyAnh_C5())
        else:
            Chosen_Proccesing(Get_XyLyAnh_C5(), list_images_c5[Get_Index_Image_C5()])

def Image_CreateMotionNoise():
    """Tạo nhiễu chuyển động"""
    try:
        output_image = CreateMotionNoise(imgin)
        c2.image(output_image, caption='Kết quả tạo nhiễu chuyển động')
    except Exception as e:
        st.error(f"Lỗi khi tạo nhiễu: {str(e)}")

def Image_DenoiseMotion():
    """Lọc nhiễu ít"""
    try:
        output_image = DenoiseMotion(imgin)
        c2.image(output_image, caption='Kết quả lọc nhiễu ít')
    except Exception as e:
        st.error(f"Lỗi khi lọc nhiễu: {str(e)}")

def Image_MoreDenoiseMotion():
    """Lọc nhiễu nhiều"""
    try:
        # Áp dụng median filter trước
        temp = cv2.medianBlur(imgin, 7)
        # Sau đó áp dụng denoise motion
        output_image = DenoiseMotion(temp)
        c2.image(output_image, caption='Kết quả lọc nhiễu nhiều')
    except Exception as e:
        st.error(f"Lỗi khi lọc nhiễu: {str(e)}")



if __name__=="__main__":
    try:
        # Cấu hình trang
        configure()
        # Hiển thị sidebar
        cs_sidebar()
        # Khởi tạo giao diện
        XuLyAnh()
        # Xử lý ảnh
        Image_All()
    except Exception as e:
        st.error(f"Có lỗi xảy ra: {str(e)}")