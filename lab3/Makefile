# Makefile (Оптимизирован для Windows 11 + Git Bash)
# Автоматически чистит логи перед каждым запуском

GAME_EXE = Build/TreasureHunt.exe

run_game:
	powershell.exe -Command "Start-Process '$(GAME_EXE)'"

run_red:
	powershell.exe -Command "Start-Process 'testClientRed.py'"

run_blue:
	powershell.exe -Command "Start-Process 'testClientBlue.py'"

# Основная команда запуска всего проекта
all: clean_logs
	powershell.exe -Command "Start-Process '$(GAME_EXE)'"
	sleep 4
	powershell.exe -Command "Start-Process 'testClientRed.py'"
	sleep 2
	powershell.exe -Command "Start-Process 'testClientBlue.py'"

# Очистка логов (безопасное удаление)
clean_logs:
	rm -f log_red.txt log_blue.txt 2>/dev/null || del /f /q log_red.txt log_blue.txt 2>nul
