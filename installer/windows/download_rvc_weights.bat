@echo off
REM Download RVC pretrained weights into third_party\rvc_core\..\assets\
setlocal enabledelayedexpansion

pushd %~dp0\..\..

set ASSETS=third_party\rvc_core\rvc_core\_vendored\assets
set BASE=https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main

if not exist "%ASSETS%\hubert"        mkdir "%ASSETS%\hubert"
if not exist "%ASSETS%\rmvpe"         mkdir "%ASSETS%\rmvpe"
if not exist "%ASSETS%\pretrained_v2" mkdir "%ASSETS%\pretrained_v2"

echo.
echo == HuBERT base (~360 MB) ==
curl -L -o "%ASSETS%\hubert\hubert_base.pt" "%BASE%/hubert_base.pt"

echo.
echo == RMVPE (~180 MB) ==
curl -L -o "%ASSETS%\rmvpe\rmvpe.pt" "%BASE%/rmvpe.pt"

REM Change to "40k" only if you want to save bandwidth.
set SRS=32k 40k 48k

for %%s in (%SRS%) do (
    echo.
    echo == Pretrained v2 %%s ==
    curl -L -o "%ASSETS%\pretrained_v2\f0G%%s.pth" "%BASE%/pretrained_v2/f0G%%s.pth"
    curl -L -o "%ASSETS%\pretrained_v2\f0D%%s.pth" "%BASE%/pretrained_v2/f0D%%s.pth"
)

echo.
echo Done. Weights are in %ASSETS%
popd
pause
