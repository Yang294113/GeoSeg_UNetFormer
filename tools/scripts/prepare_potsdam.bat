@echo off
chcp 65001 >nul
REM ======================================================
REM UNetFormer Potsdam 数据预处理 (Windows)
REM ======================================================
REM 用法:
REM   tools\scripts\prepare_potsdam.bat
REM
REM 前提: 确保 Potsdam数据集 目录在项目同级目录下
REM ======================================================

set GEOSEG_DIR=%~dp0..\..
set DATA_DIR=%GEOSEG_DIR%\..\Potsdam数据集

echo ==============================================
echo UNetFormer - Potsdam 数据预处理 (Windows)
echo GeoSeg 目录: %GEOSEG_DIR%
echo 数据集目录:  %DATA_DIR%
echo ==============================================
echo.

cd /d "%GEOSEG_DIR%"

echo [Step 1] 运行数据预处理脚本...
python tools\scripts\prepare_potsdam_data.py --data-root "%DATA_DIR%"

if %ERRORLEVEL% neq 0 (
    echo [ERROR] 数据预处理失败！
    pause
    exit /b 1
)

echo.
echo ==============================================
echo 数据预处理完成！
echo ==============================================
echo.
echo 现在你可以将整个 GeoSeg-main 文件夹上传到算力服务器。
echo 在服务器上运行:
echo   cd GeoSeg-main
echo   python train_supervision.py -c config/potsdam/unetformer.py
echo.
pause
