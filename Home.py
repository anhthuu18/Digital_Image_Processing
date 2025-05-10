from library.sidebar import * 
import streamlit as st

def Display_Info():
    # Tiêu đề chính
    st.title("🎨 PROJECT CUỐI KỲ MÔN XỬ LÝ ẢNH SỐ")

    # Tên thành viên
    st.markdown("# 👥 Thành viên thực hiện:")
    st.markdown("### Trần Huy Hạnh Phúc - 22110204")
    st.markdown("### Trương Thị Anh Thư - 22110242")

    st.markdown("---")

    st.markdown("### 📚 NỘI DUNG THỰC HIỆN")

    # Chia giao diện thành 2 cột
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### 📘 Chương 3: Xử lý hình học")
        st.markdown("- Tịnh tiến, quay, scale, affine.")
        st.markdown("- Biến đổi vị trí và kích thước ảnh.")

        st.markdown("#### 📗 Chương 4: Mức xám & Histogram")
        st.markdown("- Chuyển ảnh sang xám, âm bản.")
        st.markdown("- Cân bằng histogram, Otsu.")

        st.markdown("#### 📕 Chương 5: Lọc không gian & Biên")
        st.markdown("- Làm mờ: Trung bình, Gaussian, Median.")
        st.markdown("- Phát hiện biên: Sobel, Laplacian.")

        st.markdown("#### 🎨 Chương 9: Ảnh nghệ thuật")
        st.markdown("- Tạo hiệu ứng sketch, cartoon.")
        st.markdown("- Biến đổi phong cách ảnh.")

        st.markdown("#### 🧮 Giải phương trình bậc hai")
        st.markdown("- Nhập hệ số a, b, c.")
        st.markdown("- Tính và hiển thị nghiệm.")

    with col2:
        st.markdown("#### 🧑‍💻 Nhận dạng khuôn mặt")
        st.markdown("- 5 khuôn mặt với 2 model .onnx:")
        st.markdown("- yunet – phát hiện khuôn mặt.")
        st.markdown("- sface – trích xuất đặc trưng.")
        st.markdown("- Huấn luyện SVM để nhận dạng.")

        st.markdown("#### 🍎 Nhận dạng trái cây (YOLOv8n)")
        st.markdown("- 5 loại trái cây nhận dạng bằng YOLOv8n.")
        st.markdown("- Huấn luyện trên Google Colab.")
        st.markdown("- Triển khai hiển thị trên Streamlit.")

        st.markdown("### ♻️ Phân loại rác (YOLOv8n + JSON)")
        st.markdown("- Nhận diện 6 loại rác thải.")
        st.markdown("- Tự động hiển thị thông tin và hướng dẫn tái chế.")
        st.markdown("- Dùng file JSON để hiển thị nội dung động.")

        st.markdown("### 🅿️ Quản lý bãi đỗ xe")
        st.markdown("- Nhận diện 6 loại rác thải.")
        st.markdown("- Tự động hiển thị thông tin và hướng dẫn tái chế.")
        st.markdown("- Dùng file JSON để hiển thị nội dung động.")

    st.markdown("<br>", unsafe_allow_html=True)

def configure():
    with open("style.css") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

if __name__ == "__main__":
    configure()
    cs_sidebar()
    Display_Info()