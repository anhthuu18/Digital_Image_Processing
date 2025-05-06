import sys
import cv2
import numpy as np
from PIL import Image
import joblib
from datetime import datetime
sys.path.append('../library')
from library.bosung_streamlit.sidebar import *

# Danh sách tên người
names = ['Person1', 'Person2', 'Person3', 'Person4', 'Person5']  # Thay bằng tên thật

class FaceDetector:
    def __init__(self):
        # Load YuNet model
        self.detector = cv2.FaceDetectorYN.create(
            "pages/Source/KhuonMat/face_detection_yunet_2023mar.onnx",
            "",
            (320, 320),
            0.9,  # Score threshold
            0.3,  # NMS threshold
            5000  # Top k
        )
        
        # Load SFace model
        self.recognizer = cv2.FaceRecognizerSF.create(
            "pages/Source/KhuonMat/face_recognition_sface_2021dec.onnx",
            ""
        )
        
        # Load SVM model
        self.svc = joblib.load("pages/Source/KhuonMat/svc.pkl")

    def detect_and_recognize(self, img):
        """
        Phát hiện và nhận dạng khuôn mặt trong ảnh
        """
        height, width, _ = img.shape
        self.detector.setInputSize((width, height))
        
        # Detect faces
        faces = self.detector.detect(img)
        
        results = []
        if faces[1] is not None:
            for face in faces[1]:
                # Get face coordinates
                coords = face[:-1].astype(np.int32)
                x, y, w, h = coords[:4]
                
                # Crop and align face
                face_align = self.recognizer.alignCrop(img, face)
                
                # Get face feature
                face_feature = self.recognizer.feature(face_align)
                
                # Predict identity
                pred = self.svc.predict([face_feature])[0]
                conf = face[-1]  # Detection confidence
                
                results.append({
                    'bbox': (x, y, w, h),
                    'name': names[pred],
                    'confidence': conf
                })
        
        return results

def draw_results(img, results):
    """
    Vẽ kết quả lên ảnh
    """
    output = img.copy()
    
    for result in results:
        x, y, w, h = result['bbox']
        name = result['name']
        conf = result['confidence']
        
        # Vẽ bbox
        cv2.rectangle(output, (x, y), (x+w, y+h), (0, 255, 0), 2)
        
        # Vẽ tên và độ tin cậy
        label = f"{name} ({conf:.2f})"
        (label_w, label_h), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(output,
                    (x, y - label_h - baseline - 5),
                    (x + label_w, y),
                    (0, 255, 0),
                    cv2.FILLED)
        cv2.putText(output, label,
                   (x, y - baseline - 5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                   (0, 0, 0), 1)
    
    return output

def main():
    st.title("👥 Nhận diện khuôn mặt")
    
    # Tạo layout 2 cột
    col1, col2 = st.columns(2)
    
    # File uploader
    image_file = st.file_uploader("Upload Images", type=["jpg", "jpeg", "png"])
    
    try:
        # Khởi tạo face detector
        detector = FaceDetector()
        
        if image_file is not None:
            # Xử lý ảnh upload
            image = Image.open(image_file)
            col1.image(image, caption="Ảnh đã tải lên")
            
            # Chuyển đổi sang định dạng OpenCV
            img_array = np.array(image)
            img_array = img_array[:, :, [2, 1, 0]]  # RGB to BGR
            
            # Nhận diện khuôn mặt
            results = detector.detect_and_recognize(img_array)
            
            # Vẽ kết quả
            output_img = draw_results(img_array, results)
            
            # Hiển thị kết quả
            output_rgb = cv2.cvtColor(output_img, cv2.COLOR_BGR2RGB)
            col2.image(output_rgb, caption="Kết quả nhận diện")
            
            # Hiển thị thông tin chi tiết
            if len(results) > 0:
                st.subheader("📋 Chi tiết nhận diện")
                for i, result in enumerate(results, 1):
                    st.write(f"Người {i}:")
                    st.write(f"- Tên: {result['name']}")
                    st.write(f"- Độ tin cậy: {result['confidence']:.2%}")
        else:
            st.info("👆 Vui lòng tải lên một bức ảnh để nhận diện")
            
    except Exception as e:
        st.error(f"❌ Có lỗi xảy ra: {str(e)}")

    # Footer
    current_time = datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')
    st.markdown("---")
    st.markdown(
        f"""<div style='text-align: center'>
        <small>Current Date and Time (UTC - YYYY-MM-DD HH:MM:SS formatted): {current_time}</small><br>
        <small>Current User's Login: anhthuu18</small>
        </div>""",
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    try:
        # Cấu hình trang
        configure()
        # Hiển thị sidebar
        cs_sidebar()
        # Chạy ứng dụng
        main()
    except Exception as e:
        st.error(f"Có lỗi xảy ra: {str(e)}")