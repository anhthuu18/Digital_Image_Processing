import math
import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import cv2
import re
from PIL import Image
import easyocr
import warnings

# Tắt các cảnh báo không cần thiết từ PyTorch
warnings.filterwarnings("ignore", category=UserWarning)

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
    # Đặt giá trị mặc định để làm mới form
    if "nhap_a" in st.session_state:
        st.session_state["nhap_a"] = 0.0
    if "nhap_b" in st.session_state:
        st.session_state["nhap_b"] = 0.0
    if "nhap_c" in st.session_state:
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
        # Khởi tạo EasyOCR với ngôn ngữ tiếng Anh
        reader = easyocr.Reader(['en'], gpu=False, verbose=False)
        # Chuyển ảnh PIL thành mảng numpy
        img = np.array(image)
        # Tiền xử lý ảnh: chuyển sang ảnh xám, tăng độ tương phản, ngưỡng hóa
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        gray = cv2.convertScaleAbs(gray, alpha=2.0, beta=0)  # Tăng độ tương phản
        _, thresh = cv2.threshold(gray, 120, 255, cv2.THRESH_BINARY_INV)  # Giảm ngưỡng
        # Nhận diện văn bản
        results = reader.readtext(thresh, detail=0)
        text = " ".join(results)
        st.write(f"Văn bản thô trích xuất: {text}")  # Hiển thị văn bản thô để debug
        return text
    except Exception as e:
        st.error(f"Lỗi khi xử lý ảnh với EasyOCR: {str(e)}")
        return None

def parse_equation(text):
    if not text:
        return None, None, None
    # Loại bỏ khoảng trắng và ký tự xuống dòng
    text = text.replace("\n", "").replace(" ", "")
    # Biểu thức chính quy linh hoạt hơn, hỗ trợ cả "xn2", "x^2", "x²"
    pattern = r'([-]?\d*\.?\d*)x(?:[\^2²]|n2)([+-]?\s*\d*\.?\d*)x([+-]?\s*\d*\.?\d*)=0'
    match = re.match(pattern, text)
    if match:
        a = float(match.group(1)) if match.group(1) else 1.0
        b = float(match.group(2).replace(" ", "")) if match.group(2) else 0.0
        c = float(match.group(3).replace(" ", "")) if match.group(3) else 0.0
        st.write(f"Nhận diện: a={a}, b={b}, c={c}")  # Debug để kiểm tra hệ số
        return a, b, c
    st.warning("Không khớp với định dạng phương trình. Vui lòng kiểm tra ảnh hoặc nhập tay.")
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

            # Lưu giá trị trích xuất từ ảnh (nếu có) mà không sửa st.session_state
            if image_file is not None:
                image = Image.open(image_file)
                st.image(image, caption="Ảnh đã tải lên", use_column_width=True)
                text = extract_equation_from_image(image)
                if text:
                    a_extracted, b_extracted, c_extracted = parse_equation(text)
                    if a_extracted is not None:
                        
                        # Sử dụng giá trị trích xuất nếu người dùng không nhập tay
                        if a == 0.0 and b == 0.0 and c == 0.0:
                            a, b, c = a_extracted, b_extracted, c_extracted
                    else:
                        st.warning("Không thể nhận diện phương trình từ ảnh. Vui lòng nhập tay.")
                else:
                    st.warning("Không thể thực hiện nhận diện. Vui lòng nhập tay.")

            c1, c2 = st.columns(2)
            with c1:
                # Sửa lỗi Missing Submit Button bằng cách đảm bảo nút nằm trong form
                btn_giai = st.form_submit_button('Giải')
            with c2:
                btn_xoa = st.form_submit_button('Xóa')

            # Xử lý khi nhấn "Xóa"
            if btn_xoa:
                clear_input()
                st.experimental_rerun()  # Làm mới giao diện để cập nhật giá trị

            # Xử lý khi nhấn "Giải"
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
    Display_Info_PT()
    Main()