"""One additional model attempt for transient transport/provider failures only."""
import re
import time


def retry_model_transport(operation):
    # Budgeted clients own all requests, reservations and successful caches.
    # Never retry schema/JSON errors, authentication, or budget exhaustion.
    try:
        return operation()
    except RuntimeError as error:
        message = str(error).lower()
        transient = any(token in message for token in (
            'unexpected_eof_while_reading', 'ssleoferror', 'connection reset',
            'connection aborted', 'read timed out', 'connect timeout', 'remote end closed'))
        transient = transient or bool(re.search(r'\bhttp (429|502|503|504)\b', message))
        if not transient:
            raise
        time.sleep(2)
        return operation()
