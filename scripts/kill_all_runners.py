import os
import sys
import psutil

def main():
    print("Emergency Kill Switch: Terminating ALL Python processes on the system...")
    my_pid = os.getpid()
    
    try:
        parent_pid = psutil.Process(my_pid).ppid()
    except Exception:
        parent_pid = -1
        
    killed = 0
    for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
        try:
            pid = proc.info['pid']
            if pid == my_pid or pid == parent_pid:
                continue
                
            name = (proc.info['name'] or "").lower()
            cmd = proc.info.get('cmdline')
            cmd_str = " ".join(cmd).lower() if cmd else ""
            
            if name.startswith("py") or "python" in cmd_str:
                print(f"Terminating PID {pid} - {name} {cmd_str[:100]}...")
                proc.kill()
                killed += 1
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            pass
            
    print(f"Finished. Successfully terminated {killed} python process(es).")

if __name__ == "__main__":
    main()
