@echo off
set "SCRIPT_DIR=%~dp0"
set "ROOT_DIR=%SCRIPT_DIR%.."

cd /d "%ROOT_DIR%"
set "PYTHONPATH=%ROOT_DIR%;%PYTHONPATH%"

set "model_path=pretrained_models\SoulX-Singer\model.pt"
set "config=soulxsinger\config\soulxsinger.yaml"
set "prompt_wav_path=example\audio\en_prompt.mp3"
set "prompt_metadata_path=example\audio\en_prompt.json"
set "target_metadata_path=example\audio\en_target.json"
set "phoneset_path=soulxsinger\utils\phoneme\phone_set.json"
set "save_dir=example\generated\music"

python -m cli.inference --device cuda --model_path "%model_path%" --config "%config%" --prompt_wav_path "%prompt_wav_path%" --prompt_metadata_path "%prompt_metadata_path%" --target_metadata_path "%target_metadata_path%" --phoneset_path "%phoneset_path%" --save_dir "%save_dir%" --auto_shift --pitch_shift 0
