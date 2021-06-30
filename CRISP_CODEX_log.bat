python "CRISP_process_GUI.py" 2>&1 | mtee.exe logs\CRISP_%date:~10%%date:~4,2%%date:~7,2%-%time:~0,2%%time:~3,2%%time:~6,2%.log
PAUSE