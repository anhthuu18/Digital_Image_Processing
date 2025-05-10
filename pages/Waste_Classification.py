import os
import streamlit as st
import numpy as np
import cv2
from ultralytics import YOLO
from PIL import Image
import sys
import json

# Thêm đường dẫn thư viện nếu cần
sys.path.append('../library')
from library.sidebar import *

# Đường dẫn model PT
model_path = "pages/Source/PhanLoaiRacThai/best.pt"

# Đường dẫn file JSON
json_path = "pages/Source/PhanLoaiRacThai/waste_guide.json"

# Danh sách tên lớp
class_names = ['RacThaiKinh', 'RacThaiGiay', 'BiaCarton', 'RacHuuCo', 'RacDienTu', 'RacKimLoai']

# Danh sách ảnh mặc định
list_images_trash = [
    '',
    '1. RacThaiKinh.jpg',
    '2. RacThaiGiay.jpg',
    '3. BiaCarton.jpg',
    '4. RacHuuCo.jpg',
    '5. RacDienTu.jpg',
    '6. RacKimLoai.jpg'
]

# Đọc dữ liệu từ file JSON
try:
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        waste_types = data.get("waste_types", {})  # Lấy cấp "waste_types" từ JSON
except FileNotFoundError:
    st.error(f"Không tìm thấy file {json_path}. Vui lòng kiểm tra đường dẫn.")
    waste_types = {}
except json.JSONDecodeError:
    st.error(f"File {json_path} không đúng định dạng JSON. Vui lòng kiểm tra nội dung file.")
    waste_types = {}

# Load model
@st.cache_resource
def load_model():
    return YOLO(model_path)

def process_image(img_array, selected_class=None):
    with st.spinner('⏳ Đang phân loại...'):
        results = model.predict(
            source=img_array,
            conf=0.5,
            iou=0.4,
            verbose=False
        )[0]

        processed_img, detection_count = display_results(img_array, results, selected_class)
        processed_img_rgb = cv2.cvtColor(processed_img, cv2.COLOR_BGR2RGB)
        col2.image(processed_img_rgb, caption="📍 Kết quả phân loại", use_column_width=True)
        st.success(f"✅ Đã phát hiện {detection_count} đối tượng.")

        if detection_count > 0:
            st.subheader("📋 Chi tiết phân loại rác")
            data = []
            detected_classes = set()  # Lưu các loại rác được phát hiện
            for box, conf, cls_id in zip(
                results.boxes.xyxy.cpu().numpy(),
                results.boxes.conf.cpu().numpy(),
                results.boxes.cls.cpu().numpy().astype(int)
            ):
                if selected_class and '(' in selected_class and ')' in selected_class:
                    selected_class_name = selected_class.split('(')[1].strip(')')
                    selected_class_idx = class_names.index(selected_class_name)
                    if cls_id != selected_class_idx:
                        continue
                cls_name = class_names[cls_id] if cls_id < len(class_names) else f"Class {cls_id}"
                detected_classes.add(cls_name)
                data.append({
                    "Loại rác": cls_name,
                    "Độ tin cậy": f"{conf:.4f}"
                })
            st.table(data)

            # Hiển thị hướng dẫn tái chế cho các loại rác được phát hiện
            st.subheader("♻️ Hướng dẫn tái chế rác")
            for cls_name in detected_classes:
                if cls_name in waste_types:
                    st.markdown(f"**{cls_name}**")
                    st.write(f"- **Mô tả**: {waste_types[cls_name]['description']}")
                    st.write(f"- **Hướng dẫn tái chế**: {waste_types[cls_name]['recycling_guidance']}")
                    st.write(f"- **Phương pháp xử lý nếu không tái chế được**: {waste_types[cls_name]['disposal_method']}")
                    st.markdown("---")
                else:
                    st.warning(f"Không tìm thấy thông tin tái chế cho loại rác: {cls_name}")

def display_results(img, results, selected_class=None):
    output_img = img.copy()
    boxes = results.boxes.xyxy.cpu().numpy() if hasattr(results.boxes, 'xyxy') else []
    confs = results.boxes.conf.cpu().numpy() if hasattr(results.boxes, 'conf') else []
    cls_ids = results.boxes.cls.cpu().numpy().astype(int) if hasattr(results.boxes, 'cls') else []

    detection_count = 0
    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = map(int, box)
        cls_id = cls_ids[i]
        if selected_class and '(' in selected_class and ')' in selected_class:
            selected_class_name = selected_class.split('(')[1].strip(')')
            selected_class_idx = class_names.index(selected_class_name)
            if cls_id != selected_class_idx:
                continue
        conf = confs[i]
        cls_name = class_names[cls_id] if cls_id < len(class_names) else f"Class {cls_id}"

        cv2.rectangle(output_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        label = f"{cls_name}: {conf:.2f}"
        (label_width, label_height), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(output_img, (x1, y1 - label_height - baseline),
                     (x1 + label_width, y1), (255, 255, 255), cv2.FILLED)
        cv2.putText(output_img, label, (x1, y1 - baseline),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        detection_count += 1

    return output_img, detection_count

def detect_trash():
    st.title("🗑️ Phân loại rác với YOLOv8")
    global col1, col2, image_file
    
    # Di chuyển file uploader lên trên
    image_file = st.file_uploader("Upload Images", type=["jpg", "jpeg", "png"])
    
    # Tạo columns sau file uploader
    col1, col2 = st.columns(2)

def process_detection():
    global image_file, model
    
    selected_trash = Get_NhanDienRacThai()
    if selected_trash:  # Chỉ xử lý nếu đã chọn loại rác
        if image_file is not None:
            # Xử lý ảnh được upload
            image = Image.open(image_file)
            col1.image(image, caption="Ảnh đã tải lên", use_column_width=True)
            
            # Chuyển đổi ảnh cho xử lý
            img_array = np.array(image)
            if len(img_array.shape) == 3:
                img_array = img_array[:, :, [2, 1, 0]]  # RGB to BGR
            else:
                img_array = cv2.cvtColor(img_array, cv2.COLOR_GRAY2BGR)
            
            process_image(img_array, selected_trash)
        else:
            # Xử lý ảnh mặc định
            default_image_path = f"images/Waste_Classification/{list_images_trash[Get_Index_NhanDienRacThai()]}"
            if os.path.exists(default_image_path):
                image = Image.open(default_image_path)
                col1.image(image, caption="Ảnh mặc định", use_column_width=True)
                
                # Chuyển đổi ảnh cho xử lý
                img_array = cv2.imread(default_image_path)
                process_image(img_array, selected_trash)
            else:
                st.error(f"Không tìm thấy file ảnh: {default_image_path}")

if __name__ == "__main__":
    configure()
    cs_sidebar()
    
    # Load model
    model = load_model()
    
    # Hiển thị giao diện và xử lý
    detect_trash()
    process_detection()