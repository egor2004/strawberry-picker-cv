import sys
from PyQt6.QtWidgets import QApplication

# Импортируем класс нашего главного окна из папки gui
from gui.main_window import MainWindow

def main():
    # 1. Создаем объект приложения. 
    # sys.argv передает аргументы командной строки (если они есть)
    app = QApplication(sys.argv)
    
    # 2. Создаем экземпляр нашего главного окна
    window = MainWindow()
    
    # 3. Даем команду показать окно на экране
    window.show()
    
    # 4. Запускаем бесконечный цикл обработки событий приложения
    # sys.exit() гарантирует корректное завершение процесса при закрытии окна
    sys.exit(app.exec())

if __name__ == "__main__":
    main()