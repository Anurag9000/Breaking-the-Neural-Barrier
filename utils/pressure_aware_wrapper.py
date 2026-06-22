import sys
import time
import argparse
import subprocess
import psutil

def sample_gpu_memory_pressure(device_index=0):
    total_mib = 0
    used_mib = 0
    try:
        out = subprocess.check_output(
            ["nvidia-smi", f"--id={device_index}", "--query-gpu=memory.total,memory.used", "--format=csv,noheader,nounits"],
            stderr=subprocess.DEVNULL,
        )
        row = out.decode("utf-8").strip().splitlines()[0]
        total_text, used_text = [part.strip() for part in row.split(",", 1)]
        total_mib = int(float(total_text))
        used_mib = int(float(used_text))
    except Exception:
        pass
    if total_mib <= 0:
        return 0.0, 0
    return max(0.0, min(100.0, (float(used_mib) / float(total_mib)) * 100.0)), used_mib

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host-ram-pressure-limit-pct", type=float, default=90.0)
    parser.add_argument("--host-ram-resume-pct", type=float, default=85.0)
    parser.add_argument("--swap-pressure-limit-pct", type=float, default=100.0)
    parser.add_argument("--swap-resume-pct", type=float, default=100.0)
    parser.add_argument("--gpu-memory-pressure-limit-pct", type=float, default=90.0)
    parser.add_argument("--gpu-memory-resume-pct", type=float, default=85.0)
    parser.add_argument("--gpu-device-index", type=int, default=0)
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
    peak_paused_host_mib = 0.0
    peak_paused_gpu_mib = 0.0

    try:
        while proc.poll() is None:
            mem = psutil.virtual_memory()
            host_used_pct = mem.percent
            current_host_mib = mem.used / 1048576.0
            
            swap = psutil.swap_memory()
            swap_used_pct = swap.percent

            gpu_used_pct, current_gpu_mib = sample_gpu_memory_pressure(args.gpu_device_index)
            
            host_pressure = host_used_pct > args.host_ram_pressure_limit_pct
            swap_pressure = swap_used_pct > args.swap_pressure_limit_pct
            gpu_pressure = gpu_used_pct > args.gpu_memory_pressure_limit_pct
            
            host_resume = host_used_pct <= args.host_ram_resume_pct
            swap_resume = swap_used_pct <= args.swap_resume_pct
            gpu_resume = gpu_used_pct <= args.gpu_memory_resume_pct

            if not paused and (host_pressure or swap_pressure or gpu_pressure):
                reason = []
                if host_pressure: reason.append(f"host={host_used_pct:.2f}%>{args.host_ram_pressure_limit_pct}")
                if swap_pressure: reason.append(f"swap={swap_used_pct:.2f}%>{args.swap_pressure_limit_pct}")
                if gpu_pressure: reason.append(f"gpu={gpu_used_pct:.2f}%>{args.gpu_memory_pressure_limit_pct}")
                print(f"\n[PRESSURE] Pause requested. {' '.join(reason)}")
                try:
                    ps_proc.suspend()
                    paused = True
                    peak_paused_host_mib = current_host_mib
                    peak_paused_gpu_mib = current_gpu_mib
                except Exception as e:
                    print(f"Failed to suspend: {e}")
                
            elif paused:
                peak_paused_host_mib = max(peak_paused_host_mib, current_host_mib)
                peak_paused_gpu_mib = max(peak_paused_gpu_mib, current_gpu_mib)
                
                host_drop = peak_paused_host_mib - current_host_mib
                gpu_drop = peak_paused_gpu_mib - current_gpu_mib
                
                if (host_resume and swap_resume and gpu_resume) or host_drop >= 500.0 or gpu_drop >= 500.0:
                    print(f"\n[STATE] admission gate reopened by RAM/GPU drop or thresholds met "
                          f"host_drop={host_drop:.1f} gpu_drop={gpu_drop:.1f} "
                          f"host={host_used_pct:.2f} swap={swap_used_pct:.2f} gpu={gpu_used_pct:.2f}")
                    try:
                        ps_proc.resume()
                        paused = False
                        peak_paused_host_mib = 0.0
                        peak_paused_gpu_mib = 0.0
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
