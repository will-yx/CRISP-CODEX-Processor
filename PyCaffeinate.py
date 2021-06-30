class PyCaffeinate:
  '''Prevent OS sleep/hibernate in windows; code from:
  https://github.com/h3llrais3r/Deluge-PreventSuspendPlus/blob/master/preventsuspendplus/core.py
  API documentation:
  https://msdn.microsoft.com/en-us/library/windows/desktop/aa373208(v=vs.85).aspx'''
  ES_CONTINUOUS = 0x80000000
  ES_SYSTEM_REQUIRED = 0x00000001
  
  def __init__(self):
    pass
  
  def preventSleep(self):
    import ctypes
    ctypes.windll.kernel32.SetThreadExecutionState(PyCaffeinate.ES_CONTINUOUS|PyCaffeinate.ES_SYSTEM_REQUIRED)
  
  def allowSleep(self):
    import ctypes
    ctypes.windll.kernel32.SetThreadExecutionState(PyCaffeinate.ES_CONTINUOUS)