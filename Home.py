from library.bosung_streamlit.sidebar import *

def Display_Info():
	st.title("Xử Lý Ảnh")
	k1,k2,k3 = st.columns(3)
	k1.image(Image.open('images/einstein.jpg'))
	k2.image(Image.open('images/monalisa.jpg'))
	k3.image(Image.open('images/sunset.png'))
	t1,t2,t3 = st.columns(3)
	t1.image(Image.open('images/lena.jpg'))
	t2.image(Image.open('images/starry_night.jpg').resize((256, 256)))
	t3.image(Image.open('images/squirrel_cls.jpg').resize((256, 256)))
	
if __name__=="__main__":
	configure()
	cs_sidebar()
	Display_Info()