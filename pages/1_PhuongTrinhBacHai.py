import math
import sys
import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import cv2
import pytesseract
import re
from PIL import Image
sys.path.append('../library')
from library.sidebar import *

# Cấu hình đường dẫn đến Tesseract
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

fig, ax = plt.subplots()

def DrawPhuongTrinh(a, b, c):
    k = []
    t = []
    for x in range(-100, 100, 1):
        y = a * (x**2) + b * x + c
        k.append(x)
        t.append(y)
    ax.clear()
    ax.plot(k, t)
    ax.grid(True)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(f"Đồ thị: {a}x² + {b}x + {c} = 0")
    ax.set_xlim(-15, 10)
    ax.set_ylim(-50, 50)

def DrawPoint(x, a, b, c):
    y = a * (x**2) + b * x + c
    ax.plot(x, y, marker="o", markersize=10, markeredgecolor="red", markerfacecolor="green")

def clear_input():
    st.session_state["nhap_a"] = 0.0
    st.session_state["nhap_b"] = 0.0
    st.session_state["nhap_c"] = 0.0

def gptb2(a, b, c):
    info = []
    if a == 0:
        if b == 0:
            if c == 0:
                ket_qua = 'PTB1 có vô số nghiệm'
            else:
                ket_qua = 'PTB1 vô nghiệm'
        else:
            x = -c / b
            info.append(x)
            ket_qua = 'PTB1 có nghiệm %.2f' % x
    else:
        delta = b**2 - 4 * a * c
        if delta < 0:
            ket_qua = 'PTB2 vô nghiệm'
        else:
            x1 = (-b + math.sqrt(delta)) / (2 * a)
            info.append(x1)
            x2 = (-b - math.sqrt(delta)) / (2 * a)
            info.append(x2)
            ket_qua = 'PTB2 có nghiệm x1 = %.2f và x2 = %.2f' % (x1, x2)
    info.append(ket_qua)
    return info

def extract_equation_from_image(image):
    try:
        gray = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)
        _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)
        text = pytesseract.image_to_string(thresh, lang='eng')
        return text
    except pytesseract.TesseractNotFoundError:
        st.error("Tesseract OCR không được cài đặt hoặc không tìm thấy tại đường dẫn: C:\\Program Files\\Tesseract-OCR\\tesseract.exe. Vui lòng cài đặt Tesseract và kiểm tra lại đường dẫn.")
        return None

def parse_equation(text):
    if not text:
        return None, None, None
    text = text.replace("\n", "").replace(" ", "")
    pattern = r'([-]?\d*\.?\d*)x\^2([+-]\d*\.?\d*)x([+-]\d*\.?\d*)=0'
    match = re.match(pattern, text)
    if match:
        a = float(match.group(1)) if match.group(1) else 1.0
        b = float(match.group(2)) if match.group(2) else 0.0
        c = float(match.group(3)) if match.group(3) else 0.0
        return a, b, c
    return None, None, None

def Display_Info_PT():
    st.title("Giải phương trình bậc 2")

def Main():
    st.subheader("📤 Upload ảnh phương trình (nếu có)")
    image_file = st.file_uploader("Upload Images", type=["jpg", "jpeg", "png"])
    
    c_1, c_2 = st.columns(2)
    with c_2:
        with st.form(key='columns_in_form', clear_on_submit=False):
            a = st.number_input('Nhập a', key='nhap_a', value=0.0)
            b = st.number_input('Nhập b', key='nhap_b', value=0.0)
            c = st.number_input('Nhập c', key='nhap_c', value=0.0)

            if image_file is not None:
                image = Image.open(image_file)
                st.image(image, caption="Ảnh đã tải lên", use_column_width=True)
                text = extract_equation_from_image(image)
                if text:
                    st.write(f"Phương trình nhận diện được: {text}")
                    a_extracted, b_extracted, c_extracted = parse_equation(text)
                    if a_extracted is not None:
                        st.session_state["nhap_a"] = a_extracted
                        st.session_state["nhap_b"] = b_extracted
                        st.session_state["nhap_c"] = c_extracted
                        a, b, c = a_extracted, b_extracted, c_extracted
                    else:
                        st.warning("Không thể nhận diện phương trình từ ảnh. Vui lòng nhập tay.")
                else:
                    st.warning("Không thể thực hiện nhận diện do lỗi Tesseract. Vui lòng nhập tay.")

            c1, c2 = st.columns(2)
            with c1:
                btn_giai = st.form_submit_button('Giải')
            with c2:
                btn_xoa = st.form_submit_button('Xóa', on_click=clear_input)
            
            if btn_giai:
                s = gptb2(a, b, c)
                st.markdown('Kết quả:')
                if len(s) == 3:
                    st.markdown(s[2])
                elif len(s) == 2:
                    st.markdown(s[1])
                else:
                    st.markdown(s[0])
            else:
                st.markdown('Kết quả:')

    with c_1:
        DrawPhuongTrinh(a, b, c)
        if btn_giai and len(s) > 1:
            if len(s) == 3:
                DrawPoint(s[0], a, b, c)
                DrawPoint(s[1], a, b, c)
            elif len(s) == 2:
                DrawPoint(s[0], a, b, c)
        st.pyplot(fig)

if __name__ == "__main__":
    configure()
    cs_sidebar()
    Display_Info_PT()
    Main()