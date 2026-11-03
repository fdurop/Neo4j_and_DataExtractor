"""Short-lived progress only; graph responses remain gzip-compressed as before."""
from threading import RLock
import time
import re
from backend.routes.graph_contract import GraphError

class LoadProgress:
    def __init__(self):
        self.entries = {}
        self.lock = RLock()

    def start(self, token, owner):
        if not re.fullmatch(r"[a-f0-9]{32}", token or ""):
            raise GraphError("进度编号无效", 400)
        with self.lock:
            now=time.monotonic()
            self.entries={k:v for k,v in self.entries.items() if now-v["updated"]<180}
            if token in self.entries:
                raise GraphError("进度编号已使用，请重新加载", 409)
            if len(self.entries)>=256:
                raise GraphError("正在加载的请求过多，请稍后重试", 429)
            self.entries[token]={"owner":owner,"updated":now,"phase":"connecting"}
        return lambda **values:self.update(token,owner,**values)

    def update(self, token, owner, **values):
        with self.lock:
            item=self.entries.get(token)
            if item is not None and item["owner"]==owner:
                item.update(values,updated=time.monotonic())

    def get(self, token, owner):
        with self.lock:
            item=self.entries.get(token)
            if item is None or item["owner"]!=owner or time.monotonic()-item["updated"]>=180:
                raise GraphError("进度尚未开始或已过期", 404)
            return {k:v for k,v in item.items() if k not in {"owner","updated"}}
