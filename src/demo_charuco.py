from qt.qt_eval_charuco import Evaluation
from PyQt5.QtWidgets import QApplication
import sys


app = QApplication(sys.argv)
gui = Evaluation()
sys.exit(app.exec_())