from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton, QLabel, QVBoxLayout, QWidget
from PyQt5.QtCore import QProcess
import sys, os

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
