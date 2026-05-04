import socket
import json

class RobotTransmitter:
    def __init__(self, host="127.0.0.1", port=5005):
        """
        Инициализация UDP-передатчика.
        host: IP-адрес компьютера/контроллера робота (127.0.0.1 для тестов на одном ПК)
        port: Порт, который слушает робот
        """
        self.host = host
        self.port = port
        # AF_INET = IPv4, SOCK_DGRAM = протокол UDP
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        
    def send(self, data: dict) -> bool:
        """
        Упаковывает словарь в JSON и отправляет по сети.
        """
        try:
            # Сериализуем данные в JSON-строку и переводим в байты
            message = json.dumps(data).encode('utf-8')
            # Отправляем пакет по указанному адресу
            self.sock.sendto(message, (self.host, self.port))
            return True
        except Exception as e:
            # В реальном приложении ошибку лучше логировать, но не "ронять" программу
            print(f"Сетевая ошибка: {e}")
            return False

    def close(self):
        """Закрытие сокета при выходе из программы"""
        self.sock.close()