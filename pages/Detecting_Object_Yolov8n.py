import os
import streamlit as st
import numpy as np
import cv2
from ultralytics import YOLO
from PIL import Image
import sys
sys.path.append('../library')
from library.sidebar import *

# Đường dẫn model PT
model_path = "pages/Source/NhanDienTraiCayYolov8n/best.pt"

# Danh sách tên lớp
class_names = ['sau_rieng', 'tao', 'thanh_long', 'chuoi', 'nho']

# Danh sách ảnh mặc định
list_images_fruit = [
    '',
    '1. Sau Rieng.jpg',
    '2. Tao.jpg',
    '3. Thanh Long.jpg',
    '4. Chuoi.png',
    '5. Nho.jpg'
]

# Load model
@st.cache_resource
def load_model():
    return YOLO(model_path)

def process_image(img_array):
    with st.spinner('⏳ Đang nhận dạng...'):
        results = model.predict(
            source=img_array,
            conf=0.5,
            iou=0.4,
            verbose=False
        )[0]

        processed_img = display_results(img_array, results)
        detection_count = len(results.boxes)

        processed_img_rgb = cv2.cvtColor(processed_img, cv2.COLOR_BGR2RGB)
        st.success(f"✅ Đã phát hiện {detection_count} đối tượng.")
        col2.image(processed_img_rgb, caption="📍 Kết quả nhận dạng", use_column_width=True)

        if detection_count > 0:
            st.subheader("📋 Chi tiết nhận dạng")
            data = []
            for box, conf, cls_id in zip(
                results.boxes.xyxy.cpu().numpy(),
                results.boxes.conf.cpu().numpy(),
                results.boxes.cls.cpu().numpy().astype(int)
            ):
                cls_name = class_names[cls_id] if cls_id < len(class_names) else f"Class {cls_id}"
                data.append({
                    "Loại trái cây": cls_name,
                    "Độ tin cậy": f"{conf:.4f}"
                })
            st.table(data)

def display_results(img, results):
    output_img = img.copy()
    boxes = results.boxes.xyxy.cpu().numpy() if hasattr(results.boxes, 'xyxy') else []
    confs = results.boxes.conf.cpu().numpy() if hasattr(results.boxes, 'conf') else []
    cls_ids = results.boxes.cls.cpu().numpy().astype(int) if hasattr(results.boxes, 'cls') else []

    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = map(int, box)
        cls_id = cls_ids[i]
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

    return output_img

def detect_objects():
    st.title("🍎 Nhận dạng trái cây với YOLOv8")
    global col1, col2, image_file
    
    # Di chuyển file uploader lên trên
    image_file = st.file_uploader("Upload Images", type=["jpg", "jpeg", "png"])
    
    # Tạo columns sau file uploader
    col1, col2 = st.columns(2)

def process_detection():
    global image_file, model
    
    if Get_NhanDienTraiCay() != '':
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
            
            process_image(img_array)
        else:
            # Xử lý ảnh mặc định
            default_image_path = f"images/Detecting_Object_Yolov8n/{list_images_fruit[Get_Index_NhanDienTraiCay()]}"
            if os.path.exists(default_image_path):
                image = Image.open(default_image_path)
                col1.image(image, caption="Ảnh mặc định", use_column_width=True)
                
                # Chuyển đổi ảnh cho xử lý
                img_array = cv2.imread(default_image_path)
                process_image(img_array)
            else:
                st.error(f"Không tìm thấy file ảnh: {default_image_path}")

if __name__ == "__main__":
    configure()
    cs_sidebar()
    
    # Load model
    model = load_model()
    
    # Hiển thị giao diện và xử lý
    detect_objects()
    process_detection()
    