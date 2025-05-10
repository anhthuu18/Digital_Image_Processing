import streamlit as st
import cv2
import numpy as np
import joblib
from datetime import datetime
import av
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration

# Thiết lập trang
st.set_page_config(
    page_title="Face Recognition",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Cấu hình STUN/TURN servers
RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [
        {"urls": ["stun:stun.l.google.com:19302"]},
        {
            "urls": "turn:openrelay.metered.ca:80",
            "username": "openrelayproject",
            "credential": "openrelayproject",
        }
    ]}
)

# Khởi tạo session state để lưu danh sách người được nhận diện cho phần upload
if 'upload_recognized' not in st.session_state:
    st.session_state.upload_recognized = set()

class FaceRecognition(VideoProcessorBase):
    def __init__(self):
        self.detector = cv2.FaceDetectorYN.create(
            "pages/Source/KhuonMat/face_detection_yunet_2023mar.onnx",
            "",
            (640, 640),
            0.7,  # Giảm ngưỡng tin cậy để tăng khả năng phát hiện
            0.3,
            5000
        )
        self.recognizer = cv2.FaceRecognizerSF.create(
            "pages/Source/KhuonMat/face_recognition_sface_2021dec.onnx",
            ""
        )
        self.svc = joblib.load('pages/Source/KhuonMat/svc.pkl')
        self.encoder = joblib.load('pages/Source/KhuonMat/label_encoder.pkl')

    def process_frame_realtime(self, img):
        height, width = img.shape[:2]
        self.detector.setInputSize((width, height))
        faces = self.detector.detect(img)
        
        print(f"Realtime - Detected faces: {len(faces[1]) if faces[1] is not None else 0}")
        
        if faces[1] is not None:
            face_count = min(len(faces[1]), 5)
            for i in range(face_count):
                try:
                    face = faces[1][i]
                    coords = face[:-1].astype(np.int32)
                    confidence = face[-1]
                    face_align = self.recognizer.alignCrop(img, face)
                    face_feature = self.recognizer.feature(face_align)
                    prediction = self.svc.predict(face_feature.reshape(1, -1))[0]
                    name = self.encoder.inverse_transform([prediction])[0]
                    
                    # Vẽ bbox xung quanh khuôn mặt
                    cv2.rectangle(img, 
                                  (coords[0], coords[1]), 
                                  (coords[0]+coords[2], coords[1]+coords[3]), 
                                  (0, 255, 0),  # Màu xanh lá
                                  2)  # Độ dày viền
                    
                    # Vẽ nền cho nhãn
                    label = f"{name} ({confidence:.2f})"
                    label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
                    cv2.rectangle(img, 
                                  (coords[0], coords[1] - label_size[1] - 10),
                                  (coords[0] + label_size[0], coords[1]),
                                  (0, 255, 0),  # Màu xanh lá
                                  -1)  # Điền kín
                    
                    # Vẽ văn bản nhãn
                    cv2.putText(img, label,
                                (coords[0], coords[1] - 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                (0, 0, 0),  # Màu đen
                                2)
                except Exception as e:
                    st.error(f"Error processing face in realtime: {str(e)}")
                    print(f"Error in realtime face processing: {str(e)}")
                    continue
        return img

    def process_frame_upload(self, img):
        height, width = img.shape[:2]
        self.detector.setInputSize((width, height))
        faces = self.detector.detect(img)
        
        print(f"Upload - Detected faces: {len(faces[1]) if faces[1] is not None else 0}")
        
        if faces[1] is not None:
            face_count = min(len(faces[1]), 5)
            for i in range(face_count):
                try:
                    face = faces[1][i]
                    coords = face[:-1].astype(np.int32)
                    confidence = face[-1]
                    face_align = self.recognizer.alignCrop(img, face)
                    face_feature = self.recognizer.feature(face_align)
                    prediction = self.svc.predict(face_feature.reshape(1, -1))[0]
                    name = self.encoder.inverse_transform([prediction])[0]
                    
                    # Thêm tên vào danh sách nhận diện cho phần upload
                    st.session_state.upload_recognized.add(name)
                    
                    # Vẽ bbox xung quanh khuôn mặt
                    cv2.rectangle(img, 
                                  (coords[0], coords[1]), 
                                  (coords[0]+coords[2], coords[1]+coords[3]), 
                                  (0, 255, 0),  # Màu xanh lá
                                  2)  # Độ dày viền
                    
                    # Vẽ nền cho nhãn
                    label = f"{name} ({confidence:.2f})"
                    label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
                    cv2.rectangle(img, 
                                  (coords[0], coords[1] - label_size[1] - 10),
                                  (coords[0] + label_size[0], coords[1]),
                                  (0, 255, 0),  # Màu xanh lá
                                  -1)  # Điền kín
                    
                    # Vẽ văn bản nhãn
                    cv2.putText(img, label,
                                (coords[0], coords[1] - 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                (0, 0, 0),  # Màu đen
                                2)
                except Exception as e:
                    st.error(f"Error processing face in upload: {str(e)}")
                    print(f"Error in upload face processing: {str(e)}")
                    continue
        return img

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")
        img = self.process_frame_realtime(img)
        return av.VideoFrame.from_ndarray(img, format="bgr24")

def process_uploaded_media(file, media_type):
    processor = FaceRecognition()
    st.session_state.upload_recognized.clear()  # Xóa danh sách cũ
    
    if media_type == "image":
        image = np.array(cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR))
        if image is None:
            st.error("Không thể đọc file ảnh. Vui lòng thử file khác.")
            return None
        processed_image = processor.process_frame_upload(image)
        return processed_image
    
    elif media_type == "video":
        with open("temp_video.mp4", "wb") as f:
            f.write(file.read())
        cap = cv2.VideoCapture("temp_video.mp4")
        if not cap.isOpened():
            st.error("Không thể mở file video. Vui lòng thử file khác.")
            return None
        stframe = st.empty()
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            processed_frame = processor.process_frame_upload(frame)
            stframe.image(processed_frame, channels="BGR")
        
        cap.release()
        return None

def main():
    st.title("👥 Nhận diện khuôn mặt")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("### Camera Realtime")
        webrtc_ctx = webrtc_streamer(
            key="face-recognition",
            video_processor_factory=FaceRecognition,
            async_processing=True,
            rtc_configuration=RTC_CONFIGURATION,
            media_stream_constraints={"video": True, "audio": False},
            video_html_attrs={
                "style": {"width": "100%", "margin": "0 auto", "border": "2px solid green"},
                "controls": False,
                "autoPlay": True,
            }
        )

    with col2:
        st.write("### Nhận diện từ Video/Hình ảnh")
        media_type = st.selectbox("Chọn loại media", ["Hình ảnh", "Video"])
        file = st.file_uploader("Tải lên file", type=["jpg", "jpeg", "png"] if media_type == "Hình ảnh" else ["mp4"])
        
        if file is not None:
            media_type_lower = "image" if media_type == "Hình ảnh" else "video"
            result = process_uploaded_media(file, media_type_lower)
            if result is not None:
                st.image(result, channels="BGR", caption="Kết quả nhận diện")
            elif result is None and media_type_lower == "video":
                st.write("Video đã được xử lý.")
        
        st.write("#### Người được nhận diện:")
        if st.session_state.upload_recognized:
            for name in st.session_state.upload_recognized:
                st.write(f"- {name}")
        else:
            st.write("Chưa nhận diện được ai.")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        st.error(f"Application Error: {str(e)}")