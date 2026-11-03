"""Verified TLS and explicit database selection for the copied graph engine."""
import os
import time
from urllib.parse import urlsplit
import certifi
from neo4j import GraphDatabase as NativeGraphDatabase, TrustCustomCAs
from neo4j.exceptions import ServiceUnavailable, SessionExpired, TransientError


def run_idempotent_write(driver, query, parameters=None, attempts=3):
    """Retry only explicitly idempotent import writes, in fresh transactions.

    Each attempt consumes the result before commit. Permanent errors propagate;
    a lost commit acknowledgement can be replayed because callers use MERGE/SET.
    """
    for attempt in range(attempts):
        try:
            with driver.session() as session:
                return session.execute_write(lambda tx: tx.run(query, parameters or {}).consume())
        except (ServiceUnavailable, SessionExpired, TransientError):
            if attempt + 1 == attempts:
                raise
            time.sleep(attempt + 1)


class DatabaseDriver:
    def __init__(self, driver, database):
        self._driver = driver
        self._database = database

    def session(self, *args, **kwargs):
        if self._database:
            kwargs.setdefault("database", self._database)
        return self._driver.session(*args, **kwargs)

    def verify_connectivity(self, **kwargs):
        if self._database:
            kwargs.setdefault("database", self._database)
        return self._driver.verify_connectivity(**kwargs)

    def close(self):
        return self._driver.close()

    def __getattr__(self, name):
        return getattr(self._driver, name)


def driver_for(uri, user, password, database=None, ca_bundle=None, **kwargs):
    parsed = urlsplit(uri)
    if parsed.username or parsed.password or parsed.query or parsed.fragment:
        raise ValueError("Neo4j URI must not embed credentials or parameters")
    options = {"connection_timeout": 60, "connection_acquisition_timeout": 120, "max_transaction_retry_time": 0}
    options.update(kwargs)
    if parsed.scheme in {"neo4j+s", "bolt+s"}:
        uri = parsed._replace(scheme=parsed.scheme.removesuffix("+s")).geturl()
        options.update(encrypted=True, trusted_certificates=TrustCustomCAs(ca_bundle or certifi.where()))
    elif parsed.scheme not in {"neo4j", "bolt"} or parsed.hostname not in {"127.0.0.1", "localhost", "::1"}:
        raise ValueError("Remote Neo4j requires verified TLS (+s); plaintext is restricted to loopback")
    native = NativeGraphDatabase.driver(uri, auth=(user, password), **options)
    return DatabaseDriver(native, database)


class GraphDatabase:
    @staticmethod
    def driver(uri, auth, **kwargs):
        return driver_for(uri, *auth, database=os.getenv("APP_NEO4J_DATABASE") or None,
                          ca_bundle=os.getenv("APP_NEO4J_CA_BUNDLE") or None, **kwargs)


COURSE_SNAPSHOT_FIELDS = frozenset({"review_graph_json", "model_graph_json", "previous_review_graph_json", "review_revision", "review_updated_at", "undo_revision"})


def clean_snapshot_graph(graph):
    """Remove storage envelopes from course nodes; retain domain graph edits."""
    import copy
    result = copy.deepcopy(graph)
    for node in result.get("nodes", []):
        if "Course" in node.get("labels", []) or str(node.get("type", "")).lower() == "course":
            properties = node.get("properties", {})
            for field in COURSE_SNAPSHOT_FIELDS:
                properties.pop(field, None)
    return result
