import sys
import time
import argparse
import subprocess
import psutil

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host-ram-pressure-limit-pct", type=float, default=90.0)
    parser.add_argument("--host-ram-resume-pct", type=float, default=85.0)
    parser.add_argument("--pressure-poll-interval-sec", type=float, default=0.5)
    parser.add_argument("cmd", nargs=argparse.REMAINDER)
    args = parser.parse_args()

    cmd = args.cmd
    if cmd and cmd[0] == "--":
        cmd = cmd[1:]
    if not cmd:
        sys.exit("No command provided")

    print(f"[PRESSURE AWARE] Launching: {' '.join(cmd)}")
    proc = subprocess.Popen(cmd)
    
    try:
        ps_proc = psutil.Process(proc.pid)
    except psutil.NoSuchProcess:
        sys.exit(proc.wait())

    paused = False

    try:
        while proc.poll() is None:
            mem = psutil.virtual_memory()
            used_pct = mem.percent
            
            if not paused and used_pct > args.host_ram_pressure_limit_pct:
                print(f"\n[PRESSURE] Pause requested. host_used_pct={used_pct:.2f} > {args.host_ram_pressure_limit_pct}")
                try:
                    ps_proc.suspend()
                    paused = True
                except Exception as e:
                    print(f"Failed to suspend: {e}")
                
            elif paused and used_pct <= args.host_ram_resume_pct:
                print(f"\n[STATE] admission gate reopened by host RAM drop (foreign or paused process terminated) host_used_pct={used_pct:.2f} <= {args.host_ram_resume_pct}")
                try:
                    ps_proc.resume()
                    paused = False
                except Exception as e:
                    print(f"Failed to resume: {e}")
                
            time.sleep(max(0.1, args.pressure_poll_interval_sec))
            
    except KeyboardInterrupt:
        print("\n[INTERRUPT] Caught KeyboardInterrupt. Killing child process...")
        try:
            ps_proc.kill()
        except Exception:
            pass
        sys.exit(130)

    sys.exit(proc.returncode)

if __name__ == "__main__":
    main()
