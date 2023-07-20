import sys
import numpy as np
import matplotlib.pyplot as plt
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QLineEdit, QPushButton
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

'''
PyQt5.QtWidgets用于创建GUI窗口，matplotlib.backends.backend_qt5agg.FigureCanvasQTAgg用于将图形嵌入到Qt应用程序中。
'''


class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('Probability Function')
# 定义MainWindow类，继承QWidget类。使用setWindowTitle设置窗口的标题
        # 创建标签和文本框
        self.k_labels = []
        self.k_inputs = []
        for i in range(4):
            label = QLabel('k{}:'.format(i + 1), self)
            label.move(20, 30 + i * 40)
            self.k_labels.append(label)

            input_box = QLineEdit(self)
            input_box.move(60, 30 + i * 40)
            input_box.setFixedWidth(100)  # 设置文本框的固定宽度
            self.k_inputs.append(input_box)

        # 创建绘制按钮
        self.draw_button = QPushButton('Draw', self)
        self.draw_button.move(20, 200)
        self.draw_button.clicked.connect(self.draw)
# 创建draw_button的按钮，并设置按钮的文本为"Draw"。按钮的位置通过move方法进行设置，并将按钮的点击事件连接到self.draw方法。
        # 创建图形显示区域
        self.figure = plt.figure()
        self.canvas = FigureCanvas(self.figure)
        self.canvas.setGeometry(200, 20, self.width() - 250, self.height() - 40)
        self.canvas.setParent(self)

    def draw(self):
        # 获取用户输入的参数值
        k_values = [int(input_box.text()) for input_box in self.k_inputs]

        # d 取值范围
        d = np.linspace(0, 6000, 1000)

        # 清除之前的绘图
        self.figure.clear()

        # 绘制新的曲线
        for k in k_values:
            probability = np.exp(-d ** 2 / (2 * k ** 2))

            plt.plot(d, probability, label='Ds = {}'.format(k))

        # 图形设置
        plt.title('Probability Function')
        plt.xlabel('Distance (m)')
        plt.ylabel('Probability (P)')
        plt.xlim(0, 6000)
        plt.ylim(0, 1)
        plt.yticks(np.arange(0, 1.1, 0.1))
        plt.grid(True)
        plt.legend()

        # 更新图形显示
        self.canvas.draw()

    def resizeEvent(self, event):
        # 调整图形显示区域大小
        self.canvas.setGeometry(200, 20, self.width() - 250, self.height() - 40)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    window.setGeometry(100, 100, 800, 400)
    window.show()
    sys.exit(app.exec_())
