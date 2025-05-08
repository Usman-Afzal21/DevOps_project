@echo off
REM Windows batch script to update vector store after data changes

echo DVC hook triggered. Updating vector store and generating alerts...
python update_vector_store.py
set exit_code=%errorlevel%

if %exit_code% == 0 (
    echo Vector store updated and alerts generated successfully!
) else (
    echo Vector store update failed with exit code: %exit_code%
)

exit /b %exit_code% 