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
model_path = "pages/Source/PhanLoaiRacThai/best.pt"

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

# Dữ liệu hướng dẫn tái chế (từ JSON)
waste_types = {
    "RacThaiKinh": {
        "description": "Rác thủy tinh bao gồm chai, lọ, và các vật dụng làm từ thủy tinh.",
        "recycling_guidance": "Để tái chế thủy tinh ở Việt Nam, bạn nên rửa sạch các vật dụng thủy tinh và tách chúng ra khỏi các loại rác khác. Bạn có thể bán chúng cho người thu gom ve chai hoặc đưa đến các trung tâm tái chế nếu có. Một số nhà cung cấp bên thứ ba như VECA hoặc Ve Chai Chu Hoa cũng nhận mua thủy tinh để tái chế.",
        "disposal_method": "Nếu không thể tái chế, vui lòng vứt vào thùng rác thông thường, nhưng lưu ý rằng điều này sẽ góp phần vào rác thải ở bãi chôn lấp."
    },
    "RacThaiGiay": {
        "description": "Rác giấy bao gồm sách, vở, giấy in, bìa cứng, v.v.",
        "recycling_guidance": "Để tái chế giấy ở Việt Nam, bạn nên giữ giấy khô ráo, không dính dầu mỡ. Bạn có thể bán chúng cho người thu gom ve chai hoặc đưa đến các trung tâm tái chế nếu có. Một số công ty cũng mua giấy để tái chế.",
        "disposal_method": "Nếu không thể tái chế, vui lòng vứt vào thùng rác thông thường, nhưng lưu ý rằng điều này sẽ góp phần vào rác thải ở bãi chôn lấp."
    },
    "BiaCarton": {
        "description": "Rác bìa carton bao gồm hộp carton, bìa cứng, v.v.",
        "recycling_guidance": "Để tái chế bìa carton ở Việt Nam, bạn nên giữ bìa carton khô ráo, không dính dầu mỡ. Bạn có thể bán chúng cho người thu gom ve chai hoặc đưa đến các trung tâm tái chế nếu có. Một số nhà máy tái chế cũng mua bìa carton để xử lý.",
        "disposal_method": "Nếu không thể tái chế, vui lòng vứt vào thùng rác thông thường, nhưng lưu ý rằng điều này sẽ góp phần vào rác thải ở bãi chôn lấp."
    },
    "RacHuuCo": {
        "description": "Rác hữu cơ bao gồm thức ăn thừa, vỏ rau củ, lá cây, v.v.",
        "recycling_guidance": "Để tái chế rác hữu cơ ở Việt Nam, bạn có thể phân loại và ủ phân tại nhà hoặc đưa đến các điểm thu gom rác hữu cơ để xử lý thành phân bón. Một số tổ chức như Zero Waste Vietnam cũng khuyến khích việc composting rác hữu cơ.",
        "disposal_method": "Nếu không thể tái chế, vui lòng vứt vào thùng rác thông thường, nhưng lưu ý rằng điều này sẽ góp phần vào rác thải ở bãi chôn lấp."
    },
    "RacDienTu": {
        "description": "Rác điện tử bao gồm máy tính, điện thoại di động, tivi, tủ lạnh, v.v.",
        "recycling_guidance": "Để tái chế rác điện tử ở Việt Nam, bạn có thể tham gia chương trình Việt Nam Tái Chế, một chương trình thu hồi và tái chế rác thải điện tử miễn phí do các nhà sản xuất thiết bị điện tử khởi xướng. Bạn cũng có thể bán chúng cho các nhà tái chế hoặc đưa đến các trung tâm tái chế nếu có.",
        "disposal_method": "Nếu không thể tái chế, vui lòng không vứt vào thùng rác thông thường vì rác điện tử chứa các chất độc hại. Thay vào đó, hãy đưa đến các điểm thu gom rác điện tử hoặc tham gia các chương trình thu hồi."
    },
    "RacKimLoai": {
        "description": "Rác kim loại bao gồm lon nhôm, đồ dùng kim loại, vỏ hộp kim loại, v.v.",
        "recycling_guidance": "Để tái chế rác kim loại ở Việt Nam, bạn có thể bán chúng cho các nhà tái chế hoặc đưa đến các trung tâm tái chế nếu có. Một số công ty cũng mua kim loại để tái chế.",
        "disposal_method": "Nếu không thể tái chế, vui lòng vứt vào thùng rác thông thường, nhưng lưu ý rằng điều này sẽ góp phần vào rác thải ở bãi chôn lấp."
    }
}

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
    else:
        st.info("Vui lòng chọn loại rác từ sidebar để tiếp tục.")

if __name__ == "__main__":
    configure()
    cs_sidebar()
    
    # Load model
    model = load_model()
    
    # Hiển thị giao diện và xử lý
    detect_trash()
    process_detection()