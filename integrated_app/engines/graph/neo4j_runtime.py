"""Verified TLS and explicit database selection for the copied graph engine."""
import os
from urllib.parse import urlsplit
import certifi
from neo4j import GraphDatabase as NativeGraphDatabase, TrustCustomCAs


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
    options = {"connection_timeout": 8, "connection_acquisition_timeout": 15, "max_transaction_retry_time": 0}
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
