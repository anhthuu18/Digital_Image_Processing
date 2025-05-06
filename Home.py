from library.bosung_streamlit.sidebar import *
from PIL import Image
import streamlit as st

def Display_Info():
    # Tiêu đề chính với biểu tượng
    st.title("🎨 Project Cuối kỳ môn Xử Lý Ảnh Số")

    # Tên thành viên
    st.markdown("#### 👥 Thành viên thực hiện:")
    st.markdown("- Trần Huy Hạnh Phúc - 22110204")
    st.markdown("- Trương Thị Anh Thư - 22110242")

    # Dòng phân cách
    st.markdown("---")

    # Hàng 1: Danh nhân & nghệ thuật
    st.markdown("### 🧠 Danh nhân và Nghệ thuật")
    c1, c2, c3 = st.columns(3)

    with c1:
        st.image(Image.open('images/einstein.jpg'), use_column_width=True)
        st.markdown("**Albert Einstein**  \nHình ảnh của nhà khoa học vĩ đại.")

    with c2:
        st.image(Image.open('images/monalisa.jpg'), use_column_width=True)
        st.markdown("**Mona Lisa**  \nTác phẩm nghệ thuật nổi tiếng của Leonardo da Vinci.")

    with c3:
        st.image(Image.open('images/lena.jpg'), use_column_width=True)
        st.markdown("**Lena**  \nHình ảnh tiêu chuẩn trong xử lý ảnh số.")

    st.markdown("<br>", unsafe_allow_html=True)

    # Hàng 2: Tác phẩm - Thiên nhiên - Động vật
    st.markdown("### 🌄 Thiên nhiên và nghệ thuật")
    c4, c5, c6 = st.columns(3)

    with c4:
        st.image(Image.open('images/starry_night.jpg').resize((256, 256)), use_column_width=True)
        st.markdown("**Starry Night**  \nBức tranh kinh điển của Vincent van Gogh.")

    with c5:
        st.image(Image.open('images/sunset.png'), use_column_width=True)
        st.markdown("**Hoàng hôn**  \nPhong cảnh thiên nhiên yên bình.")

    with c6:
        st.image(Image.open('images/squirrel_cls.jpg').resize((256, 256)), use_column_width=True)
        st.markdown("**Sóc**  \nHình ảnh minh họa phân loại động vật.")

    st.markdown("<br>", unsafe_allow_html=True)
    # CSS tùy chỉnh
def configure():
    with open("style.css") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

if __name__ == "__main__":
    configure()
    cs_sidebar()
    Display_Info()
