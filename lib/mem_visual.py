from PyQt5.QtWidgets import QApplication, QLabel, QWidget, QListWidget, QGridLayout
from PyQt5.QtCore import QTimer
import sys
import time
import threading
import os, psutil


class mem_thread(threading.Thread):
    def __init__(self, pid, mode):
        threading.Thread.__init__(self)
        self.pid = pid
        self.mode = mode

    def run(self):
        print("here")        
        app = QApplication(sys.argv)
        mine_ram = ramlabel(pid = self.pid)
        mine_ram.resize(200,100)
        mine_ram.show()
        app.exec_()




class ramlabel(QWidget):
    def __init__(self,parent = None, pid = None):
        super(ramlabel,self).__init__(parent)
        self.pid = pid
        self.setWindowTitle("Ram usage")
        self.listFile = QListWidget()
        self.label = QLabel('333')

        layout = QGridLayout()

        self.timer = QTimer()

        layout.addWidget(self.label, 0,0,3,5)
        self.timer.timeout.connect(self.show_mem)
        self.setLayout(layout)
        self.timer.start(1000)

    def show_mem(self):
        process = psutil.Process(self.pid)
        self.label.setText(f"{process.memory_info().rss/(1024*1024*1024)} GB\n {psutil.virtual_memory()[2]}%")
        with open("ramout.txt","a") as f:
            f.write(f"{psutil.virtual_memory()[2]}\n")


