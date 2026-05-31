import socket
import threading
import json
import time

HOST = "127.0.0.1"
PORT = 8080

def receive_messages(sock):
    buffer = ""
    try:
        while True:
            data = sock.recv(65536).decode("utf-8", errors="ignore")
            if not data:
                print("СОЕДИНЕНИЕ ПОТЕРЯНО")
                break

            buffer += data
            while "\n" in buffer:
                line, buffer = buffer.split("\n", 1)
                line = line.strip()
                if not line: continue

                print(f"\n[Данные]: {line}")

    except Exception as e:
        print(f"Ошибка: {e}")
    finally:
        sock.close()

def main():
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        sock.connect((HOST, PORT))
        print(f"ПОДКЛЮЧЕНО К {HOST}:{PORT} (РЕЖИМ СЛУШАТЕЛЯ)")
        
        t = threading.Thread(target=receive_messages, args=(sock,), daemon=True)
        t.start()

        while t.is_alive():
            time.sleep(1)

    except Exception as e:
        print(f"НЕ УДАЛОСЬ ПОДКЛЮЧИТЬСЯ: {e}")
    finally:
        sock.close()

if __name__ == "__main__":
    main()