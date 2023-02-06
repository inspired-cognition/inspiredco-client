from __future__ import annotations

import logging
from typing import Any, final

import requests


class ClientBase:
    """A base class for clients."""

    _api_key: str
    _session: requests.Session
    _logger: logging.Logger

    def __init__(self, api_key: str):
        """Initialize the client.

        Args:
            api_key: The API key to use.
        """
        self._api_key = api_key
        self._session = requests.Session()
        self._logger = logging.getLogger("inspiredco-client")

    @final
    @property
    def api_key(self) -> str:
        """Returns the API key."""
        return self._api_key

    @final
    def http_post(
        self, url: str, json: Any, headers: dict[str, str] | None = None
    ) -> requests.Response:
        """Send a POST request authenticated with the API key.

        Args:
            url: The URL to send the request to.
            json: The JSON body to send.
            headers: The headers to send, or empty if none.

        Returns:
            The response.
        """
        modified_headers = {
            "Authorization": f"Bearer {self._api_key}",
            **(headers or {}),
        }
        return self._session.post(url, json=json, headers=modified_headers)
