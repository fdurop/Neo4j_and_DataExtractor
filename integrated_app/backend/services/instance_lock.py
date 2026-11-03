"""OS-held single-instance lock; automatically released even after a crash."""
from pathlib import Path
import os


class InstanceLock:
    def __init__(self, data_dir):
        self.path = Path(data_dir) / ".instance.lock"
        self.handle = None

    def __enter__(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.handle = self.path.open("a+b")
        self.handle.seek(0, 2)
        if self.handle.tell() == 0:
            self.handle.write(b"0")
            self.handle.flush()
        self.handle.seek(0)
        try:
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(self.handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(self.handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            self.handle.close()
            raise RuntimeError("该运行目录已有服务实例，请使用已有服务或独立数据目录") from None
        return self

    def __exit__(self, *args):
        if self.handle:
            self.handle.close()
