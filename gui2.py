import io
import os
import pickle
import re
import sys
from io import BytesIO

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import QByteArray
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import QFileDialog
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier


def resource_path(relative_path):
    # For PyInstaller bundled app, this retrieves the correct path for bundled files
    try:
        base_path = sys._MEIPASS  # This is the temporary path where PyInstaller stores bundled files
    except Exception:
        base_path = os.path.abspath(".")  # For normal script execution
    return os.path.join(base_path, relative_path)


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.setWindowTitle("Attack Classifier")
        MainWindow.resize(700, 400)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        
        # File Import
        self.frame = QtWidgets.QFrame(self.centralwidget)
        self.frame.setGeometry(QtCore.QRect(10, 10, 281, 151))
        
        self.pushButton = QtWidgets.QPushButton("Import Log File", self.frame)
        self.pushButton.setGeometry(QtCore.QRect(90, 60, 101, 21))
        self.pushButton.clicked.connect(self.open_file_dialog)
        
        self.label = QtWidgets.QLabel(self.frame)
        self.label.setGeometry(QtCore.QRect(30, 90, 221, 41))
        self.label.setWordWrap(True)
        self.label.setText("<html><head/><body><p align=\"center\">No file selected</p></body></html>")

        self.label_2 = QtWidgets.QLabel(self.frame)
        self.label_2.setGeometry(QtCore.QRect(10, 11, 261, 20))
        self.label_2.setObjectName("label_2")
        self.label_2.setText("<html><head/><body><p><span style=\" font-size:9pt; font-weight:600;\">1. Import logs (txt format)</span></p></body></html>")
        
        self.processButton = QtWidgets.QPushButton(">", self.frame)
        self.processButton.setGeometry(QtCore.QRect(254, 60, 21, 23))
        self.processButton.setEnabled(False)
        self.processButton.clicked.connect(self.process_logs)

        # Progress Bar
        self.frame_2 = QtWidgets.QFrame(self.centralwidget)
        self.frame_2.setGeometry(QtCore.QRect(300, 10, 391, 151))
        self.frame_2.setObjectName("frame_2")
        
        self.label_3 = QtWidgets.QLabel(self.frame_2)
        self.label_3.setGeometry(QtCore.QRect(10, 10, 201, 20))
        self.label_3.setObjectName("label_3")
        self.label_3.setText("<html><head/><body><p><span style=\" font-size:9pt; font-weight:600;\">2. Model processing...</span></p></body></html>")
        
        self.progressBar = QtWidgets.QProgressBar(self.frame_2)
        self.progressBar.setGeometry(QtCore.QRect(20, 60, 351, 21))
        self.progressBar.setProperty("value", 0)
        self.progressBar.setObjectName("progressBar")
        
        # Output
        self.frame_3 = QtWidgets.QFrame(self.centralwidget)
        self.frame_3.setGeometry(QtCore.QRect(10, 170, 681, 221))
        self.textEdit = QtWidgets.QTextEdit(self.frame_3)
        self.textEdit.setGeometry(QtCore.QRect(20, 40, 281, 171))
        self.textEdit.setReadOnly(True)
        
        self.label_4 = QtWidgets.QLabel(self.frame_3)
        self.label_4.setGeometry(QtCore.QRect(10, 10, 291, 20))
        self.label_4.setObjectName("label_4")
        self.label_4.setText("<html><head/><body><p><span style=\" font-size:9pt; font-weight:600;\">3. Output</span></p></body></html>")
        
        self.imageLabel = QtWidgets.QLabel(self.frame_3)
        self.imageLabel.setGeometry(QtCore.QRect(330, 40, 341, 171))
        self.imageLabel.setStyleSheet("border: 1px solid black;")
        self.imageLabel.setAlignment(QtCore.Qt.AlignCenter)

        self.comboBox = QtWidgets.QComboBox(self.frame_3)
        self.comboBox.setGeometry(QtCore.QRect(330, 10, 191, 21))
        self.comboBox.setObjectName("comboBox")
        self.comboBox.addItems(["Heatmap", "Trend Chart"])
        self.comboBox.currentIndexChanged.connect(self.update_plot)
        
        self.imageButton = QtWidgets.QPushButton("View Full Image", self.frame_3)
        self.imageButton.setGeometry(QtCore.QRect(530, 10, 130, 21))
        self.imageButton.clicked.connect(self.open_full_image)
        self.imageButton.setEnabled(False)

        MainWindow.setCentralWidget(self.centralwidget)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)
        
        # Load Model and Vectorizer
        model_path = resource_path("xgboost_attack_model.pkl")
        vectorizer_path = resource_path("vectorizer.pkl")

        with open(model_path, "rb") as f:
            self.model = pickle.load(f)
        with open(vectorizer_path, "rb") as f:
            self.vectorizer = pickle.load(f)
    
    def open_file_dialog(self):
        try:
            file_path, _ = QFileDialog.getOpenFileName(None, "Select Log File", "", "Text Files (*.txt);;All Files (*)")
            if file_path:
                if os.path.exists(file_path) and os.path.getsize(file_path) > 0:
                    self.label.setText(f"<html><head/><body><p align=\"center\">{file_path}</p></body></html>")
                    self.file_path = file_path
                    self.processButton.setEnabled(True)
                else:
                    self.textEdit.setText("File is empty")
                    self.processButton.setEnabled(False)
        except Exception as e:
            QtWidgets.QMessageBox.critical(None, "Error", f"Failed to open file: {str(e)}")

    
    def process_logs(self):
        self.progressBar.setValue(10)
        QtWidgets.QApplication.processEvents()
        
        df = self.parse_logs_fixed([self.file_path])
        self.progressBar.setValue(30)
        QtWidgets.QApplication.processEvents()
        
        df = self.label_logs(df)
        self.progressBar.setValue(50)
        QtWidgets.QApplication.processEvents()
        
        X_new = self.vectorizer.transform(df['message']).toarray()
        self.progressBar.setValue(70)
        QtWidgets.QApplication.processEvents()
        
        df['predicted_attack_type'] = self.model.predict(X_new)

        self.df = df  

        self.textEdit.setText(df.to_string())
        self.progressBar.setValue(90)
        QtWidgets.QApplication.processEvents()
        
        self.display_heatmap(df)
        self.progressBar.setValue(100)
        QtWidgets.QApplication.processEvents()
    
    def parse_logs_fixed(self, log_files):
        data = []
        for file in log_files:
            with open(file, 'r') as f:
                for line in f:
                    cleaned_line = re.sub(r'\x1b\[[0-9;]*[a-zA-Z]', '', line)
                    timestamp = re.search(r'\[\d{8}\s+\d{2}:\d{2}:\d{2}\]', cleaned_line)
                    service = re.search(r'\]\s+(\w+)', cleaned_line)
                    message = re.search(r'(/dionaea/.*)', cleaned_line)
                    if timestamp and service and message:
                        data.append({
                            "timestamp": timestamp.group(0).strip("[]"),
                            "service": service.group(1),
                            "message": message.group(1)
                        })
        return pd.DataFrame(data)

    def label_logs(self, df):
        def label(row):
            service = row['service'].lower()
            message = row['message'].lower()

            if 'http' in service or 'http' in message:
                if 'sql' in message or 'injection' in message:
                    return 'SQL Injection'
                elif 'xss' in message or 'script' in message:
                    return 'Cross-Site Scripting'
                else:
                    return 'HTTP Attack'
            elif 'ftp' in service or 'ftp' in message:
                return 'FTP Attack'
            elif 'sip' in service or 'pptp' in service or 'pptp' in message:
                return 'SIP Attack'
            elif 'ssh' in service or 'brute' in message:
                return 'SSH Attack'
            elif 'dns' in service or 'dns' in message:
                return 'DNS Attack'
            elif 'log_sqlite' in service or 'log_sqlite' in message:
                return 'Log_SQLite Attack'
            elif 'download' in message or 'malware' in message:
                return 'Malware Download'
            else:
                return 'Other Attack'
            
        df['attack_type'] = df.apply(label, axis=1)
        return df
    
    def update_plot(self):
        selected_option = self.comboBox.currentText()
        
        if selected_option == "Heatmap":
            self.display_heatmap(self.df)
        elif selected_option == "Trend Chart":
            self.display_trend_chart(self.df)

    def display_heatmap(self, df):
        attack_counts = df['attack_type'].value_counts().reset_index()
        attack_counts.columns = ['Attack Type', 'Count']
        heatmap_data = attack_counts.pivot_table(index="Attack Type", values="Count")

        fig, ax = plt.subplots(figsize=(10, 6))
        sns.heatmap(heatmap_data, annot=True, fmt=".0f", cmap="coolwarm", linewidths=0.5, ax=ax)

        ax.set_title("Heatmap of Attack Types in Logs")
        ax.set_xlabel("Frequency")
        ax.set_ylabel("Attack Type")

        plt.tight_layout()

        buf = io.BytesIO()
        fig.savefig(buf, format="png", bbox_inches='tight', dpi=100)
        buf.seek(0)

        byte_array = QByteArray(buf.read())
        image = QImage.fromData(byte_array)
        self.full_pixmap = QPixmap.fromImage(image)

        scaled_pixmap = self.full_pixmap.scaled(self.imageLabel.width(), self.imageLabel.height(), QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation)
        self.imageLabel.setPixmap(scaled_pixmap)

        self.imageButton.setEnabled(True)

    def display_trend_chart(self, df):
        df["timestamp"] = pd.to_datetime(df["timestamp"], format="%d%m%Y %H:%M:%S")
        df["date"] = df["timestamp"].dt.date

        attack_trends = df.groupby(["date", "attack_type"]).size().reset_index(name="count")

        fig, ax = plt.subplots(figsize=(14, 7))
        sns.barplot(x="date", y="count", hue="attack_type", data=attack_trends, palette="tab10", ax=ax)

        ax.set_title("Attack Type Trends Over Time", fontsize=16)
        ax.set_xlabel("Date", fontsize=14)
        ax.set_ylabel("Count of Attacks", fontsize=14)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
        ax.legend(title="Attack Type", bbox_to_anchor=(1.05, 1), loc="upper left")
        ax.grid(axis="y", linestyle="--", alpha=0.7)

        plt.tight_layout()

        buf = io.BytesIO()
        fig.savefig(buf, format="png", bbox_inches='tight', dpi=100)
        buf.seek(0)

        byte_array = QByteArray(buf.read())
        image = QImage.fromData(byte_array)
        self.full_pixmap = QPixmap.fromImage(image)

        scaled_pixmap = self.full_pixmap.scaled(self.imageLabel.width(), self.imageLabel.height(), QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation)
        self.imageLabel.setPixmap(scaled_pixmap)

    def open_full_image(self):
        if not hasattr(self, "full_pixmap"):
            return

        self.image_window = QtWidgets.QWidget()
        self.image_window.setWindowTitle("Full Image View")
        self.image_window.resize(self.full_pixmap.width(), self.full_pixmap.height())

        layout = QtWidgets.QVBoxLayout()
        label = QtWidgets.QLabel()
        label.setPixmap(self.full_pixmap)
        layout.addWidget(label)

        self.image_window.setLayout(layout)
        self.image_window.show()


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())
