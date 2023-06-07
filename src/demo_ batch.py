from qt.qt_batch_eval import EvaluationBatch
from PyQt5.QtWidgets import QApplication
import sys

app = QApplication(sys.argv)
gui = EvaluationBatch()
sys.exit(app.exec_())
