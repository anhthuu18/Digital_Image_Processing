import sys
from PIL import Image
import cv2
import numpy as np
sys.path.append('../library')
from library.sidebar import *
from library.Chapter03 import *

list_images_c3 = ['','1. Negative Image.png','2. Logarit Image.png','3. LuyThua.png','4. BienDoiTuyenTinhTungPhan.png','5. Histogram.png','6. Can Bang Histogram.png','7. Can Bang Histogram Anh Mau.png','8. Local Histogram.png','9. Thong Ke Histogram.png','10. Loc Box.png','11. Loc Gauss.png','12. PhanNguong.png','13. Loc_Median.png','14. Shapen.png','15. Gradient.png']

def XuLyAnh():
    global image_file
    global c1,c2
    st.title("Xử lý ảnh - "+Get_XyLyAnh_C3())
    image_file = st.file_uploader("Upload Images", type=["bmp", "png","jpg","jpeg"])
    c1,c2 = st.columns(2)

imgin = None
def InsertImage(image_file,path=''):
    global image
    if image_file is not None:
        image = Image.open(image_file)
        c1.image(image, caption=None)
    else:
        image = Image.open("./images/ImageProcessing_Chapter3/"+path)
        c1.image(image, caption=None)
        image.close()

def Chosen_Proccesing(option,path=''):
    global imgin
    if image_file is not None:
        # Mở lại file để xử lý
        image = Image.open(image_file)
        image_array = np.array(image)
        # Chuyển sang grayscale nếu là ảnh màu
        if len(image_array.shape) == 3:
            if option == '7. Cân bằng Histogram của ảnh màu.':
                imgin = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)
            else:
                imgin = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
        else:
            imgin = image_array
        image.close()
    else:
        InsertImage(image_file,list_images_c3[Get_Index_Image()])
        if option == '7. Cân bằng Histogram của ảnh màu.':
            imgin = cv2.imread("./images/ImageProcessing_Chapter3/"+path, cv2.IMREAD_COLOR)
        else:
            imgin = cv2.imread("./images/ImageProcessing_Chapter3/"+path, cv2.IMREAD_GRAYSCALE)

    if(option == '1. Negative Image.'):
        Image_Negative()
    elif(option == '2. Lograrit ảnh.'):
        Image_Logarit()
    elif(option == '3. Lũy thừa ảnh.'):
        Image_Power()
    elif(option == '4. Biến đổi tuyến tính từng phần.'):
        Image_PiecewiseLinear()
    elif(option == '5. Histogram'):
        Image_Histogram()
    elif(option == '6. Cân bằng Histogram'):
        Image_HistEqual()
    elif(option == '7. Cân bằng Histogram của ảnh màu.'):
        Image_HistEqualColor(imgin)
    elif(option == '8. Local Histogram.'):
        Image_LocalHist()
    elif(option == '9. Thống kê Histogram'):
        Image_HistStat()    
    elif(option == '10. Lọc Box'):
        Image_MyBoxFilter()
    elif(option == '11. Lọc Gauss'):
        Image_onLowpassGauss()
    elif(option == '12. Phân Ngưỡng'):
        Image_Threshold()
    elif(option == '13. Lọc Median'):
        Image_MedianFilter()
    elif(option == '14. Sharpen'):
        Image_Sharpen()
    elif(option == '15. Gradient'):
        Image_Gradient()

# Các hàm xử lý ảnh giữ nguyên
def Image_Negative():
    output_image = Negative(imgin)
    c2.image(output_image, caption=None)

def Image_Logarit():
    output_image = Logarit(imgin)
    c2.image(output_image, caption=None)

def Image_Power():
    output_image = Power(imgin)
    c2.image(output_image, caption=None)

def Image_PiecewiseLinear():
    output_image = PiecewiseLinear(imgin)
    c2.image(output_image, caption=None)

def Image_Histogram():
    output_image = Histogram(imgin)
    c2.image(output_image, caption=None)

def Image_HistEqual():
    output_image = HistEqual(imgin)
    c2.image(output_image, caption=None)

def Image_HistEqualColor(image_temp):
    output_image = HistEqualColor(image_temp)
    c2.image(output_image, caption=None)

def Image_LocalHist():
    output_image = LocalHist(imgin)
    c2.image(output_image, caption=None)

def Image_HistStat():
    output_image = HistStat(imgin)
    c2.image(output_image, caption=None)

def Image_MyBoxFilter():
    output_image = MyBoxFilter(imgin)
    c2.image(output_image, caption=None)

def Image_MedianFilter():
    output_image = MedianFilter(imgin)
    c2.image(output_image, caption=None)

def Image_Threshold():
    output_image = Threshold(imgin)
    c2.image(output_image, caption=None)

def Image_Sharpen():
    output_image = Sharpen(imgin)
    c2.image(output_image, caption=None)

def Image_Gradient():
    output_image = Gradient(imgin)
    c2.image(output_image, caption=None)

def Image_onLowpassGauss():
    output_image = cv2.GaussianBlur(imgin,(43,43),7.0)
    c2.image(output_image, caption=None)

def Image_All():
    global image_file
    if(Get_XyLyAnh_C3()!=''):
        if(image_file is not None):
            InsertImage(image_file)
            Chosen_Proccesing(Get_XyLyAnh_C3())
        else:
            Chosen_Proccesing(Get_XyLyAnh_C3(),list_images_c3[Get_Index_Image()])

if __name__=="__main__":
    configure()
    cs_sidebar()
    XuLyAnh()
    Image_All()
