import sys
from PIL import Image
import cv2
import numpy as np
sys.path.append('../library')
from library.bosung_streamlit.sidebar import *
from library.bosung_streamlit.Chapter04 import *


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
		image = Image.open(image_file) # You must transfome your picture into grey
		c1.image(image, caption=None)
		# Chuyển sang cv2 để dùng sau này
		frame = np.array(image)
		frame = frame[:, :, [2, 1, 0]] # BGR -> RGB
		image.close()
	else:
		image = Image.open("./images/ImageProcessing_Chapter4/"+path) # You must transfome your picture into grey
		c1.image(image, caption=None)
		image.close()

def Chosen_Proccesing(option,path=''):
	global imgin
	if(option != '3. Vẽ bộ lọc note-reject'):
		if(image_file is not None):
			path=image_file.name
		else:
			InsertImage(image_file,list_images_c4_0[Get_Index_Image_C4_0()])
		imgin = cv2.imread("./images/ImageProcessing_Chapter4/"+path,cv2.IMREAD_GRAYSCALE)

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
			# Mo de xu ly
			InsertImage(image_file)
			#Chosen_Proccesing(Get_XyLyAnh_C4_0())
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