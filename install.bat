cd /d "%~dp0"

python -m venv venv

call venv\Scripts\activate.bat

python -m pip install --upgrade pip
python -m pip install -r requirements.txt

python main.py

echo Готово.
pause