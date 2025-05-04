import math
import sys
import tkinter
sys.path.append('../library')
from library.bosung_streamlit.sidebar import *
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
fig, ax = plt.subplots()
def DrawPhuongTrinh(a,b,c):
	k=[]
	t=[]
	for x in range(-100,100,1):
		y = a*(x**2) + b * x + c
		k.append(x)
		t.append(y)
	global fig
	global ax
	ax.plot(k,t)

def DrawPoint(x,a,b,c):
	plt.plot(x, a*(x**2) + b * x + c, marker="o", markersize=5, markeredgecolor="red", markerfacecolor="green")

def clear_input():
	st.session_state["nhap_a"] = 0.0
	st.session_state["nhap_b"] = 0.0
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
			x = -c/b
			info.append(x)
			ket_qua = 'PTB1 có nghiệm %.2f' % x
	else:
		delta = b**2 - 4*a*c
		if delta < 0:
			ket_qua = 'PTB2 vô nghiệm'
		else:
			x1 = (-b + math.sqrt(delta))/(2*a)
			info.append(x1)
			x2 = (-b - math.sqrt(delta))/(2*a)
			info.append(x2)
			ket_qua = 'PTB2 có nghiệm x1 = %.2f và x2 = %.2f' % (x1, x2)
	info.append(ket_qua)
	return info
def Display_Info_PT():
	st.title("Giải phương trình bậc 2")
def Main():
	#..2 columns..
	a = 0
	b = 0
	c = 0
	c_1, c_2 = st.columns(2)
	with c_2:
		with st.form(key='columns_in_form', clear_on_submit = False):
			a = st.number_input('Nhập a', key = 'nhap_a')
			b = st.number_input('Nhập b', key = 'nhap_b')
			c = st.number_input('Nhập c', key = 'nhap_c')
			c1, c2 = st.columns(2)
			with c1:
				btn_giai = st.form_submit_button('Giải')
			with c2:
				btn_xoa = st.form_submit_button('Xóa', on_click=clear_input)
			if btn_giai:
				s = gptb2(a, b, c)
				if(len(s)==3):
					st.markdown('Kết quả: ' + s[2])
					DrawPoint(s[0],a,b,c)
					DrawPoint(s[1],a,b,c)
				elif(len(s)==2):
					st.markdown('Kết quả: ' + s[1])
					DrawPoint(s[0],a,b,c)
				else:
					st.markdown('Kết quả: ' + s[0])
			else:
				st.markdown('Kết quả:')
	with c_1:	
		DrawPhuongTrinh(a,b,c)
		st.pyplot(fig)
		
if __name__=="__main__":
	configure()
	cs_sidebar()
	Display_Info_PT()
	Main()