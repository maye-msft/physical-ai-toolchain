"""Resolve a Hugging Face token from Azure Key Vault inside an Azure ML job.

Azure ML jobs expose the compute identity through a per-job MSI endpoint rather
than VM IMDS. AmlCompute sets ``MSI_ENDPOINT`` and ``MSI_SECRET``; Kubernetes
compute sets only ``MSI_ENDPOINT``, served by the job's identity sidecar. The
token is printed to stdout without a trailing newline so the caller can capture
it with command substitution.
"""

# cspell:ignore clientid

from __future__ import annotations

import json
import os
import sys
import urllib.parse
import urllib.request
from collections.abc import Mapping, Sequence
from typing import Protocol

KEY_VAULT_RESOURCE = "https://vault.azure.net"
MSI_API_VERSION = "2017-09-01"
KEY_VAULT_API_VERSION = "7.4"
REQUEST_TIMEOUT_SECONDS = 30


class _Opener(Protocol):
    def open(self, request: urllib.request.Request, timeout: float) -> object: ...


def _append_query(url: str, query: str) -> str:
    separator = "&" if "?" in url else "?"
    return f"{url}{separator}{query}"


def build_identity_request(environ: Mapping[str, str]) -> urllib.request.Request:
    """Build the Key Vault access-token request for the Azure ML job identity."""
    msi_endpoint = environ.get("MSI_ENDPOINT")
    if not msi_endpoint:
        raise RuntimeError(
            "MSI_ENDPOINT is required to read HF_TOKEN_SECRET_URL; submit the job with a managed identity"
        )
    query = {"api-version": MSI_API_VERSION, "resource": KEY_VAULT_RESOURCE}
    client_id = environ.get("AZURE_CLIENT_ID") or environ.get("DEFAULT_IDENTITY_CLIENT_ID")
    if client_id:
        query["clientid"] = client_id
    msi_secret = environ.get("MSI_SECRET")
    headers = {"secret": msi_secret} if msi_secret else {}
    return urllib.request.Request(_append_query(msi_endpoint, urllib.parse.urlencode(query)), headers=headers)


def build_secret_request(secret_url: str, access_token: str) -> urllib.request.Request:
    """Build the Key Vault get-secret request."""
    return urllib.request.Request(
        _append_query(secret_url, f"api-version={KEY_VAULT_API_VERSION}"),
        headers={"Authorization": f"Bearer {access_token}"},
    )


def _read_json(opener: _Opener, request: urllib.request.Request) -> dict[str, object]:
    with opener.open(request, timeout=REQUEST_TIMEOUT_SECONDS) as response:
        payload = json.load(response)
    if not isinstance(payload, dict):
        raise RuntimeError(f"unexpected response from {urllib.parse.urlsplit(request.full_url).netloc}")
    return payload


def resolve_token(
    environ: Mapping[str, str],
    *,
    identity_opener: _Opener | None = None,
    key_vault_opener: _Opener | None = None,
) -> str:
    """Return the Hugging Face token stored at ``HF_TOKEN_SECRET_URL``."""
    secret_url = environ.get("HF_TOKEN_SECRET_URL")
    if not secret_url:
        raise RuntimeError("HF_TOKEN_SECRET_URL is required")
    # The identity endpoint is local to the job and must bypass any configured proxy.
    identity_opener = identity_opener or urllib.request.build_opener(urllib.request.ProxyHandler({}))
    key_vault_opener = key_vault_opener or urllib.request.build_opener()

    access_token = _read_json(identity_opener, build_identity_request(environ)).get("access_token")
    if not isinstance(access_token, str) or not access_token:
        raise RuntimeError("Azure ML job identity returned no access token")

    secret_value = _read_json(key_vault_opener, build_secret_request(secret_url, access_token)).get("value")
    if not isinstance(secret_value, str) or not secret_value:
        raise RuntimeError("Hugging Face token secret is empty")
    return secret_value


def main(argv: Sequence[str] | None = None) -> int:
    if argv:
        print(f"usage: {sys.argv[0]} (configured through environment variables)", file=sys.stderr)
        return 2
    try:
        token = resolve_token(os.environ)
    except (RuntimeError, OSError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1
    sys.stdout.write(token)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
