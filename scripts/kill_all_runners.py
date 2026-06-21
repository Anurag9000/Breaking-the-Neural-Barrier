import os
import sys
import psutil

def main():
    print("Emergency Kill Switch: Scanning for running MLPS python processes...")
    my_pid = os.getpid()
    killed = 0
    for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
        try:
            cmd = proc.info.get('cmdline')
            if not cmd:
                continue
            cmd_str = " ".join(cmd).lower()
            if proc.info['pid'] == my_pid:
                continue
            
            # Identify target processes that belong to our MLPS/tabular pipeline
            if "python" in (proc.info['name'] or "").lower() or "python" in cmd_str:
                if "mlps" in cmd_str or "dae_dnn" in cmd_str or "run_stl" in cmd_str or "run_goliath" in cmd_str or "run_task" in cmd_str:
                    print(f"Terminating PID {proc.info['pid']} - {cmd_str[:100]}...")
                    proc.kill()
                    killed += 1
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            pass
            
    print(f"Finished. Successfully terminated {killed} orphan/running process(es).")

if __name__ == "__main__":
    main()
