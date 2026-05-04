import socket
import json

UDP_IP = "127.0.0.1"
UDP_PORT = 5005

# Создаем принимающий сокет
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind((UDP_IP, UDP_PORT))

print(f"[КОНТРОЛЛЕР РОБОТА] Запущен и слушает порт {UDP_PORT}...")
print("Ожидание координат от системы технического зрения...\n")

while True:
    try:
        data, addr = sock.recvfrom(65535) # Получаем пакет
        payload = json.loads(data.decode('utf-8'))
        
        # Если в кадре есть спелые ягоды, готовые к срезу
        for det in payload.get("detections", []):
            if det["actionable"]:
                cut = det["keypoints"]["cut_point"]
                print(f"ПРИНЯТА КОМАНДА: X:{cut['x']:.1f}, Y:{cut['y']:.1f}")
    except KeyboardInterrupt:
        break