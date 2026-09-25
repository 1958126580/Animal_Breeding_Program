@echo off
rem ABP launcher for cmd.exe: delegates to abp.ps1 (UTF-8, logging, exit status).
rem Usage: abp.cmd run "D:\育种 数据\analysis.toml" --out "D:\育种 数据\结果"
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0abp.ps1" %*
exit /b %ERRORLEVEL%
