## library
import os
import sys
from tkinter import *
## main gui
root =Tk()
root.title('Stopy')
root.geometry('640x480')
## fungsi script
def bca():
    os.system('bbca.py')
def bri():
    os.system('bbri.py')
def bni():
    os.system('bbni.py')
def mega():
    os.system('mega.py')
def mandiri():
    os.system('mandiri.py')

## GUI
l1 = Label(root, text='Prediksi Saham Menggunakan Metode LSTM')
l1.pack()
l2 = Label(root, text='Pilihan Bank : ')
b1 = Button(root, text='Bank BCA', command=bca)
b1.pack()
b2 = Button(root, text='Bank BRI', command=bri)
b2.pack()
b3 = Button(root, text='Bank BNI', command=bni)
b3.pack()
b4 = Button(root, text='Bank MEGA', command=mega)
b4.pack()
b5 = Button(root, text='Bank Mandiri', command=mandiri)
b5.pack()


root.mainloop()