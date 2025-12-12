REM Launches Jupyter Notebook for the GLP-1 NLP demo project
REM Activates the project environment and opens notebooks at repo root

@echo off

REM Get directory of this script
set PROJECT_ROOT=%~dp0

REM Conda environment name
set ENV_NAME=glp1_nlp

call conda activate %ENV_NAME%

cd /d %PROJECT_ROOT%

jupyter notebook

pause
