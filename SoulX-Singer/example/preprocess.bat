@echo off
set "SCRIPT_DIR=%~dp0"
:: Remove trailing backslash
set "SCRIPT_DIR=%SCRIPT_DIR:~0,-1%"
:: Get parent directory
for %%I in ("%SCRIPT_DIR%\..") do set "ROOT_DIR=%%~fI"

cd /d "%ROOT_DIR%" || exit /b
set "PYTHONPATH=%ROOT_DIR%;%PYTHONPATH%"

set "device=cuda"

::::::: Run Prompt Annotation :::::::
set "audio_path=example/audio/zh_prompt.mp3"
set "save_dir=example/transcriptions/zh_prompt"
set "language=English"
set "vocal_sep=False"
set "max_merge_duration=30000"

python -m preprocess.pipeline ^
    --audio_path "%audio_path%" ^
    --save_dir "%save_dir%" ^
    --language %language% ^
    --device %device% ^
    --vocal_sep %vocal_sep% ^
    --max_merge_duration %max_merge_duration%


::::::: Run Target Annotation :::::::
set "audio_path=../DataSynthesizer/Gay or European/Gay or European.mp3"
set "save_dir=../DataSynthesizer/Gay or European"
set "language=English"
set "vocal_sep=False"
set "max_merge_duration=60000"

python -m preprocess.pipeline ^
    --audio_path "%audio_path%" ^
    --save_dir "%save_dir%" ^
    --language %language% ^
    --device %device% ^
    --vocal_sep %vocal_sep% ^
    --max_merge_duration %max_merge_duration%
