"""Provenance-preserving HTTPS fetcher.

Network access is explicit at the caller. The requested URL is never silently
rewritten. Redirect destinations are revalidated against the same HTTPS policy
on every hop; redirect chains and response bytes are recorded locally.
"""

from __future__ import annotations

import hashlib
import ipaddress
import socket
import time
from dataclasses import dataclass
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import HTTPRedirectHandler, Request, build_opener

from ..control.codec import canonical_bytes, digest
from ..control.errors import InputError, IntegrityError
from ..control.files import atomic_write, ensure_private_dir
from .sources import Source

FETCH_SCHEMA = "ablagent.research.fetch.v1"
MAX_BYTES = 16 * 1024 * 1024


def _validate_url(url: str) -> None:
    parsed = urlparse(url)
    if parsed.scheme != "https" or parsed.username or parsed.password or not parsed.hostname:
        raise InputError("only credential-free https URLs are allowed")
    try:
        addresses = socket.getaddrinfo(parsed.hostname, parsed.port or 443, type=socket.SOCK_STREAM)
    except OSError as exc:
        raise InputError(f"source host cannot be resolved: {parsed.hostname}") from exc
    for address in addresses:
        ip = ipaddress.ip_address(address[4][0])
        if ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_reserved or ip.is_multicast:
            raise InputError(f"source host resolves to a disallowed address: {ip}")


class _CheckedRedirects(HTTPRedirectHandler):
    def __init__(self, chain: list[str]):
        super().__init__()
        self.chain = chain

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        _validate_url(newurl)
        self.chain.append(newurl)
        return super().redirect_request(req, fp, code, msg, headers, newurl)


@dataclass(frozen=True)
class FetchReceipt:
    schema: str
    source_id: str
    requested_url: str
    final_url: str
    redirects: tuple[str, ...]
    retrieved_at: str
    status: int
    content_sha256: str
    bytes: int
    license: str
    receipt_digest: str = ""


def fetch_source(
    source: Source,
    output_dir: Path,
    *,
    timeout: float = 15.0,
    user_agent: str = "ablagent-research/1",
) -> FetchReceipt:
    source.validate()
    _validate_url(source.url)
    output_dir = ensure_private_dir(Path(output_dir), create=not Path(output_dir).exists())
    request = Request(source.url, headers={"User-Agent": user_agent, "Accept": "*/*"})
    redirects: list[str] = []
    opener = build_opener(_CheckedRedirects(redirects))
    try:
        with opener.open(request, timeout=timeout) as response:
            final = response.geturl()
            _validate_url(final)
            data = response.read(MAX_BYTES + 1)
            status = getattr(response, "status", 200)
    except (HTTPError, URLError, TimeoutError) as exc:
        raise IntegrityError(f"source fetch failed for {source.source_id}: {exc}") from exc
    if len(data) > MAX_BYTES:
        raise IntegrityError(f"source response exceeds {MAX_BYTES} bytes")
    content_sha = hashlib.sha256(data).hexdigest()
    blob = output_dir / f"{source.source_id}-{content_sha}.bin"
    atomic_write(blob, data)
    body = {
        "schema": FETCH_SCHEMA,
        "source_id": source.source_id,
        "requested_url": source.url,
        "final_url": final,
        "redirects": redirects,
        "retrieved_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "status": status,
        "content_sha256": content_sha,
        "bytes": len(data),
        "license": source.license,
    }
    receipt = dict(body)
    receipt["receipt_digest"] = digest(body)
    atomic_write(
        output_dir / f"{source.source_id}-{content_sha}.receipt.json", canonical_bytes(receipt)
    )
    return FetchReceipt(**receipt)
