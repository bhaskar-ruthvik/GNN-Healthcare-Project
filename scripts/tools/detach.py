"""Run a command fully detached from the launching app: double fork + setsid (new session and process group, reparented to
launchd), stdin from /dev/null, stdout/stderr appended to a log. Usage: python3 scripts/tools/detach.py <log> <cmd> [args...]
Prints the PID of the detached command."""
import os, sys, pathlib  # bootstrap: make referral_gnn importable from any scripts/ subfolder (or `pip install -e .`)
sys.path.insert(0, str(next(p for p in pathlib.Path(__file__).resolve().parents if (p / "referral_gnn" / "paths.py").is_file())))
from referral_gnn.paths import workspace, script_path, code_file  # noqa: E402,F401
import os, sys
log, cmd = sys.argv[1], sys.argv[2:]
r, w = os.pipe()
if os.fork():                                   # parent: wait for the grandchild's pid, print it, exit
    os.close(w); print(os.read(r, 64).decode().strip()); sys.exit(0)
os.close(r); os.setsid()
if os.fork(): os._exit(0)
os.write(w, str(os.getpid()).encode()); os.close(w)
fd = os.open(log, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644); os.dup2(fd, 1); os.dup2(fd, 2)
os.dup2(os.open(os.devnull, os.O_RDONLY), 0)
os.execvp(cmd[0], cmd)
