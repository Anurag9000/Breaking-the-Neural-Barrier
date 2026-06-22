import csv
import sys
import time
from pathlib import Path
from typing import Dict, List, Any, Optional

class ContinuousLogger:
    def __init__(self, results_dir: Path, model_name: str, mode: str, resume: bool = False):
        self.results_dir = results_dir
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.model_name = model_name
        self.mode = mode
        
        self.log_file = self.results_dir / "training_log.txt"
        self.csv_file = self.results_dir / "training_stats.csv"
        file_mode = "a" if resume else "w"

        self.txt_handle = open(self.log_file, file_mode, encoding="utf-8")

        # CSV handling
        self.csv_handle = open(self.csv_file, file_mode, newline="", encoding="utf-8")
        self.csv_writer = None 
        # We don't init writer yet because we need to know columns from first log call
        
        self.log_console(f"Initialized ContinuousLogger for {model_name} (Mode: {mode})")
        self.log_console(f"Logs: {self.log_file}")
        self.log_console(f"CSV: {self.csv_file}")

    def log_console(self, message: str):
        """Log to console AND text file with timestamp."""
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        formatted = f"[{timestamp}] {message}"
        
        # Print to console
        print(formatted, flush=True)
        
        # Write to file and flush immediately
        self.txt_handle.write(formatted + "\n")
        self.txt_handle.flush()

    def log_epoch_stats(self, stats: Dict[str, Any]):
        """
        Log epoch statistics to CSV.
        Expected stats keys: epoch, width, depth, train_loss, val_loss, etc.
        """
        # Add timestamp and metadata
        stats["timestamp"] = time.strftime("%Y-%m-%d %H:%M:%S")
        stats["model"] = self.model_name
        stats["mode"] = self.mode
        
        # Initialize CSV writer if first write
        if self.csv_writer is None:
            fieldnames = list(stats.keys())
            # Ensure commonly first columns are first
            priority = ["timestamp", "model", "mode", "epoch", "width", "depth", "train_loss", "val_loss"]
            fieldnames.sort(key=lambda x: priority.index(x) if x in priority else 999)
            
            self.csv_writer = csv.DictWriter(self.csv_handle, fieldnames=fieldnames, extrasaction="ignore")
            
            # Write header only if file is empty
            if self.csv_file.stat().st_size == 0:
                self.csv_writer.writeheader()
                self.csv_handle.flush()
        
        self.csv_writer.writerow(stats)
        self.csv_handle.flush() # CRITICAL: Ensure data is saved immediately

    def close(self):
        if self.txt_handle:
            self.txt_handle.close()
        if self.csv_handle:
            self.csv_handle.close()

def setup_logging(results_dir: Path, model_name: str, args: Any) -> ContinuousLogger:
    """Helper to setup logging from args."""
    mode = getattr(args, "adp_mode", "standard")
    return ContinuousLogger(results_dir, model_name, mode)

if hasattr(sys, "excepthook"):
    _original_excepthook = sys.excepthook
    def _emergency_kill_excepthook(exctype, value, traceback):
        if issubclass(exctype, KeyboardInterrupt):
            print("\n[INTERRUPT] Caught KeyboardInterrupt system-wide. Triggering emergency kill switch...")
            try:
                import subprocess
                repo_root = Path(__file__).resolve().parent.parent
                kill_script = repo_root / "scripts" / "kill_all_runners.py"
                if kill_script.exists():
                    subprocess.run([sys.executable, str(kill_script)])
            except Exception:
                pass
            sys.exit(130)
        _original_excepthook(exctype, value, traceback)
    sys.excepthook = _emergency_kill_excepthook

