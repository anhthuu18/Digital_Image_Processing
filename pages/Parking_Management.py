import streamlit as st
from streamlit_drawable_canvas import st_canvas
import numpy as np
import cv2
import pickle
from ultralytics import YOLO
from PIL import Image
import time

# Cấu hình page
st.set_page_config(
    page_title="Parking Detection System",
    page_icon="🚗",
    layout="wide"
)

# Load CSS từ file với encoding UTF-8
def load_css(css_file):
    with open(css_file, 'r', encoding='utf-8') as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

# Load CSS
    load_css('style.css')


# Constants and Configuration
MODEL_PATH = "pages/Source/QuanLyBaiDoXe/best.pt"
CLASS_NAMES = ['car']
CONF_THRESHOLD = 0.5
IOU_THRESHOLD = 0.4

# Load model
@st.cache_resource
def load_model():
    return YOLO(MODEL_PATH)

# Session state initialization
if "parking_spaces" not in st.session_state:
    try:
        with open("pages/Source/QuanLyBaiDoXe/parking_spaces.pkl", "rb") as f:
            st.session_state["parking_spaces"] = pickle.load(f)
    except FileNotFoundError:
        st.session_state["parking_spaces"] = []

def compute_iou(box1, box2):
    x1, y1, x2, y2 = box1
    x1_, y1_, x2_, y2_ = box2
    xi1 = max(x1, x1_)
    yi1 = max(y1, y1_)
    xi2 = min(x2, x2_)
    yi2 = min(y2, y2_)
    inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
    box1_area = (x2 - x1) * (y2 - y1)
    box2_area = (x2_ - x1_) * (y2_ - y1_)
    union_area = box1_area + box2_area - inter_area
    return inter_area / union_area if union_area != 0 else 0

def process_detection(img_array, model):
    with st.spinner('🔄 Đang xử lý...'):
        results = model.predict(
            source=img_array,
            conf=CONF_THRESHOLD,
            iou=IOU_THRESHOLD,
            verbose=False
        )[0]

        processed_img = img_array.copy()
        car_boxes = results.boxes.xyxy.cpu().numpy()
        occupied_spaces = 0

        for i, space in enumerate(st.session_state["parking_spaces"]):
            occupied = False
            for car_box in car_boxes:
                iou = compute_iou([space[0][0], space[0][1], space[1][0], space[1][1]], car_box)
                if iou > 0.5:
                    occupied = True
                    occupied_spaces += 1
                    break
                    
            color = (0, 0, 255) if occupied else (0, 255, 0)
                
            overlay = processed_img.copy()
            cv2.rectangle(overlay, space[0], space[1], color, -1)
            cv2.addWeighted(overlay, 0.3, processed_img, 0.7, 0, processed_img)
            cv2.rectangle(processed_img, space[0], space[1], color, 2)
            
            label_pos = (space[0][0] + 5, space[0][1] + 25)
            cv2.putText(processed_img, f"{i+1}", label_pos,
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        total_spaces = len(st.session_state["parking_spaces"])
        num_cars = len(car_boxes)
        empty_spaces = total_spaces - occupied_spaces

        return processed_img, num_cars, total_spaces, empty_spaces

def main():
    # Header
    st.markdown("<h1>🚗 HỆ THỐNG GIÁM SÁT BÃI ĐỖ XE THÔNG MINH</h1>", unsafe_allow_html=True)
    
    # Load model
    model = load_model()
    
    # Main content
    st.markdown("<div class='upload-section'>", unsafe_allow_html=True)
    img_file_buffer = st.file_uploader("📤 Tải lên ảnh bãi đỗ xe", 
                                     type=["jpg", "jpeg", "png", "bmp"],
                                     help="Hỗ trợ các định dạng: JPG, JPEG, PNG, BMP")
    st.markdown("</div>", unsafe_allow_html=True)
    
    if img_file_buffer is not None:
        # Process uploaded image
        nparr = np.frombuffer(img_file_buffer.read(), np.uint8)
        img_array = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        display_img = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)

        # Create two columns
        col1, col2 = st.columns([1, 1])

        # Left column
        with col1:
            st.markdown("### 📸 Khu vực vẽ vùng đỗ xe")
            canvas_result = st_canvas(
                fill_color="rgba(0, 255, 0, 0.3)",
                stroke_width=2,
                background_image=Image.fromarray(display_img),
                update_streamlit=True,
                height=display_img.shape[0],
                width=display_img.shape[1],
                drawing_mode="rect",
                key="canvas",
            )

            if st.button("💾 Lưu vùng đỗ xe"):
                if canvas_result.json_data and canvas_result.json_data["objects"]:
                    st.session_state["parking_spaces"] = []
                    for obj in canvas_result.json_data["objects"]:
                        left = int(obj["left"])
                        top = int(obj["top"])
                        width = int(obj["width"])
                        height = int(obj["height"])
                        st.session_state["parking_spaces"].append(
                            ((left, top), (left + width, top + height))
                        )
                    with open("parking_spaces.pkl", "wb") as f:
                        pickle.dump(st.session_state["parking_spaces"], f)
                    st.success("✅ Đã lưu vùng đỗ xe thành công!")

        # Right column
        with col2:
            st.markdown("### 🔍 Kết quả phân tích")
            if st.button("🔄 Bắt đầu nhận diện"):
                if st.session_state["parking_spaces"]:
                    processed_img, num_cars, total_spaces, empty_spaces = process_detection(img_array, model)
                    
                    # Display processed image
                    st.image(cv2.cvtColor(processed_img, cv2.COLOR_BGR2RGB), 
                            use_column_width=True,
                            caption="Kết quả nhận diện")
                    
                    # Calculate occupancy rate
                    occupancy_rate = (total_spaces - empty_spaces) / total_spaces * 100 if total_spaces > 0 else 0
                    
                    # Display system info and statistics
                    st.markdown(f"""

                    🚗 Số xe
                    {num_cars}

                    🅿️ Tổng chỗ đỗ
                    {total_spaces}

                    ✨ Còn trống
                    {empty_spaces}

                    📊 Tỷ lệ sử dụng
                    {occupancy_rate:.1f}%
                    """)
                else:
                    st.warning("⚠️ Vui lòng vẽ và lưu vùng đỗ xe trước khi nhận diện!")

if __name__ == "__main__":
    main()