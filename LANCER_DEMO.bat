@echo off
G:
cd "G:\Mon Drive\travail final\FULL_BACKUP_RAPIDE\datascientest_projet"

echo Nettoyage des fichiers temporaires...
for /d /r . %%d in (__pycache__) do @if exist "%%d" rd /s /q "%%d"

echo Recherche de Conda...
set CONDA_PATH=%USERPROFILE%\anaconda3\Scripts\activate.bat
if not exist "%CONDA_PATH%" set CONDA_PATH=C:\anaconda3\Scripts\activate.bat
if not exist "%CONDA_PATH%" set CONDA_PATH=%USERPROFILE%\Miniconda3\Scripts\activate.bat

if exist "%CONDA_PATH%" (
    echo Activation de l'environnement via %CONDA_PATH%
    call "%CONDA_PATH%" base
) else (
    echo [ATTENTION] Anaconda n'a pas ete trouve dans les dossiers standards.
)

echo Verification/Creation de l'environnement Python...
call conda create -n rakuten_env python=3.10 -y
call conda activate rakuten_env

echo Installation/Mise a jour des dependances...
pip install -r requirements.txt

echo Lancement du Dashboard Rakuten...
streamlit run src/streamlit/app.py

pause