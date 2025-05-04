import streamlit as st
from pathlib import Path
import base64
from PIL import Image
import sys
sys.path.append('./library/streamlit_extras')
from streamlit_extras.switch_page_button import switch_page


st.set_page_config(
     page_title='Digital Image Processing',
     layout="wide",
     initial_sidebar_state="expanded",
)

def configure():
	st.markdown("""
		<style>
			[class="contract-trigger"]{
				font-size:24;
			}
			[data-testid="stSidebarNav"]{
				display:None;
			}
		</style>
	""", unsafe_allow_html=True)
eventlist_1=['','1. Negative Image.','2. Lograrit ảnh.','3. Lũy thừa ảnh.','4. Biến đổi tuyến tính từng phần.','5. Histogram','6. Cân bằng Histogram','7. Cân bằng Histogram của ảnh màu.','8. Local Histogram.','9. Thống kê Histogram','10. Lọc Box','11. Lọc Gauss','12. Phân Ngưỡng','13. Lọc Median','14. Sharpen','15. Gradient']
eventlist_4_0=['','1. Spectrum','2. Lọc trong miền tần số','3. Vẽ bộ lọc note-reject','4. Xóa nhiễu Moire']
eventlist_9=['','1. PhiLeGa','2. HatGao']
def img_to_bytes(img_path):
    img_bytes = Path(img_path).read_bytes()
    encoded = base64.b64encode(img_bytes).decode()
    return encoded

def Get_XyLyAnh_C3():
    return chosen_event
def Get_XyLyAnh_C4_0():
    return chosen_event1
def Get_XyLyAnh_C9():
    return chosen_event2


def Get_Index_Image():
    return eventlist_1.index(chosen_event)
def Get_Index_Image_C4_0():
    return eventlist_4_0.index(chosen_event1)
def Get_Index_Image_C9():
    return eventlist_9.index(chosen_event2)

chosen_event_old = ''
def switch_page_images_C3(chosen_event):
    global chosen_event_old
    if (chosen_event==''):
        return
    if (chosen_event=='1. Negative Image.' and chosen_event_old!=chosen_event):
        chosen_event_old = chosen_event
        switch_page('Image_Processing_Chapter3')
    if (chosen_event=='2. Lograrit ảnh.' and chosen_event_old!=chosen_event):
        chosen_event_old = chosen_event
        switch_page('Image_Processing_Chapter3')
    if (chosen_event=='3. Lũy thừa ảnh.' and chosen_event_old!=chosen_event):
        chosen_event_old = chosen_event
        switch_page('Image_Processing_Chapter3')
    if (chosen_event=='4. Biến đổi tuyến tính từng phần.' and chosen_event_old!=chosen_event):
        chosen_event_old = chosen_event
        switch_page('Image_Processing_Chapter3')
    if (chosen_event=='5. Histogram' and chosen_event_old!=chosen_event):
        chosen_event_old = chosen_event
        switch_page('Image_Processing_Chapter3')
    if (chosen_event=='6. Cân bằng Histogram' and chosen_event_old!=chosen_event):
        chosen_event_old = chosen_event
        switch_page('Image_Processing_Chapter3')
    if (chosen_event=='7. Cân bằng Histogram của ảnh màu.' and chosen_event_old!=chosen_event):
        chosen_event_old = chosen_event
        switch_page('Image_Processing_Chapter3')
    if (chosen_event=='8. Local Histogram.' and chosen_event_old!=chosen_event):
        chosen_event_old = chosen_event
        switch_page('Image_Processing_Chapter3')
    if (chosen_event=='9. Thống kê Histogram' and chosen_event_old!=chosen_event):
        chosen_event_old = chosen_event
        switch_page('Image_Processing_Chapter3')
    if (chosen_event=='10. Lọc Box' and chosen_event_old!=chosen_event):
        chosen_event_old = chosen_event
        switch_page('Image_Processing_Chapter3')
    if (chosen_event=='11. Lọc Gauss' and chosen_event_old!=chosen_event):
        chosen_event_old = chosen_event
        switch_page('Image_Processing_Chapter3')
    if (chosen_event=='12. Phân Ngưỡng' and chosen_event_old!=chosen_event):
        chosen_event_old = chosen_event
        switch_page('Image_Processing_Chapter3')
    if (chosen_event=='13. Lọc Median' and chosen_event_old!=chosen_event):
        chosen_event_old = chosen_event
        switch_page('Image_Processing_Chapter3')
    if (chosen_event=='14. Sharpen' and chosen_event_old!=chosen_event):
        chosen_event_old = chosen_event
        switch_page('Image_Processing_Chapter3')
    if (chosen_event=='15. Gradient' and chosen_event_old!=chosen_event):
        chosen_event_old = chosen_event
        switch_page('Image_Processing_Chapter3')

chosen_event_old_C9 = ''
def switch_page_images_C9(chosen_event):
    global chosen_event_old_C9
    if (chosen_event==''):
        return
    if (chosen_event=='1. PhiLeGa' and chosen_event_old_C9!=chosen_event):
        chosen_event_old_C9 = chosen_event
        switch_page('Image_Processing_Chapter9')
    if (chosen_event=='2. HatGao' and chosen_event_old_C9!=chosen_event):
        chosen_event_old_C9 = chosen_event
        switch_page('Image_Processing_Chapter9')

# chosen_event_old_C5 = ''
# def switch_page_images_C5(chosen_event):
#     global chosen_event_old_C5
#     if (chosen_event==''):
#         return
#     if (chosen_event=='1. Tạo Nhiễu Ảnh' and chosen_event_old_C5!=chosen_event):
#         chosen_event_old_C5 = chosen_event
#         switch_page('Image_Processing - C5')
#     if (chosen_event=='2. Lọc Ảnh Ít Nhiễu' and chosen_event_old_C5!=chosen_event):
#         chosen_event_old_C5 = chosen_event
#         switch_page('Image_Processing - C5')
#     if (chosen_event=='3. Vẽ bộ lọc note-reject' and chosen_event_old_C5!=chosen_event):
#         chosen_event_old_C5 = chosen_event
#         switch_page('Image_Processing - C5')
#     if (chosen_event=='4. Xóa nhiễu Moire' and chosen_event_old_C5!=chosen_event):
#         chosen_event_old_C5 = chosen_event
#         switch_page('Image_Processing - C5')

chosen_event_old_C4_0 = ''
def switch_page_images_C4_0(chosen_event):
    global chosen_event_old_C4_0
    if (chosen_event==''):
        return
    if (chosen_event=='1. Spectrum' and chosen_event_old_C4_0!=chosen_event):
        chosen_event_old_C4_0 = chosen_event
        switch_page('Image_Processing_Chapter4')
    if (chosen_event=='2. Lọc trong miền tần số' and chosen_event_old_C4_0!=chosen_event):
        chosen_event_old_C4_0 = chosen_event
        switch_page('Image_Processing_Chapter4')
    if (chosen_event=='3. Lọc Ảnh Nhiều Nhiễu' and chosen_event_old_C4_0!=chosen_event):
        chosen_event_old_C4_0 = chosen_event
        switch_page('Image_Processing_Chapter4')
    if (chosen_event=='4. Xóa nhiễu Moire' and chosen_event_old_C4_0!=chosen_event):
        chosen_event_old_C4_0 = chosen_event
        switch_page('Image_Processing_Chapter4')

def cs_sidebar():
    st.sidebar.markdown('''<img src='data:image/png;base64,{}' class='img-fluid' width=250 height=300>'''.format(img_to_bytes("logo.png")), unsafe_allow_html=True)
    st.sidebar.header('Digital Image Processing')
    st.sidebar.markdown('''Các Nội Dung:''', unsafe_allow_html=True)
    if st.sidebar.button('1. Giải phương trình bậc 2') :
    	switch_page('PhuongTrinhBacHai')
    if st.sidebar.button('2. Nhận diện 5 khuôn mặt') :
    	switch_page('Detecting_5_faces')
    if st.sidebar.button('3. Nhận diện đối tượng yolov8n') :
    	switch_page('Detecting_Object_Yolov8n')
    global chosen_event
    chosen_event = st.sidebar.selectbox('6. Xử lý ảnh _ Chương 3', eventlist_1)
    if(chosen_event!=''):
        switch_page_images_C3(chosen_event)
    
    global chosen_event1
    chosen_event1 = st.sidebar.selectbox('7. Xử lý ảnh _ Chương 4',eventlist_4_0)
    if(chosen_event1!=''):
        switch_page_images_C4_0(chosen_event1)

    global chosen_event2
    chosen_event2 = st.sidebar.selectbox('9. Xử lý ảnh _ Chương 9',eventlist_9)
    if(chosen_event2!=''):
        switch_page_images_C9(chosen_event2)
    # st.sidebar.header('Các bài tập làm thêm')

    # if st.sidebar.button('1. Hand Tracking Basic.'):
    #     switch_page('HandTracking_BaSic_LT')
    # if st.sidebar.button('2. Dem So Ngon Tay.'):
    #     switch_page('Finger_Counter_LT')
    st.sidebar.markdown('''<hr>''', unsafe_allow_html=True)
    st.sidebar.markdown('''<small>[22110242 - 22110204]</small>''', unsafe_allow_html=True)
