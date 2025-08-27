# File: livekit/plugins/yandex/auth.py

import os
import time
import asyncio
import aiohttp
import logging
from typing import Optional, Dict, Any
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class AuthConfig:
    """Configuration for Yandex Cloud authentication."""
    api_key: Optional[str] = None
    folder_id: Optional[str] = None
    iam_token: Optional[str] = None
    iam_token_expires_at: Optional[float] = None
    
    @classmethod
    def from_env(cls) -> "AuthConfig":
        """Create auth config from environment variables."""
        return cls(
            api_key=os.environ.get("YANDEX_API_KEY"),
            folder_id=os.environ.get("YANDEX_FOLDER_ID"),
        )
    
    def validate(self) -> None:
        """Validate that required authentication credentials are present."""
        if not self.api_key and not self.iam_token:
            raise ValueError(
                "Either YANDEX_API_KEY or IAM token must be provided. "
                "Set the YANDEX_API_KEY environment variable."
            )
        
        if not self.folder_id:
            raise ValueError(
                "YANDEX_FOLDER_ID environment variable must be set."
            )

class YandexAuth:
    """Helper class for Yandex Cloud authentication."""
    
    def __init__(self, config: Optional[AuthConfig] = None):
        self.config = config or AuthConfig.from_env()
        self.config.validate()
    
    def get_headers(self) -> Dict[str, str]:
        """Get authentication headers for API requests."""
        headers = {"Content-Type": "application/json"}
        
        if self.config.api_key:
            headers["Authorization"] = f"Api-Key {self.config.api_key}"
        elif self.config.iam_token:
            if self._is_token_expired():
                logger.warning("IAM token has expired")
            headers["Authorization"] = f"Bearer {self.config.iam_token}"
        
        return headers
    
    def get_folder_id(self) -> str:
        """Get folder ID."""
        return self.config.folder_id
    
    def _is_token_expired(self) -> bool:
        """Check if IAM token has expired."""
        if not self.config.iam_token_expires_at:
            return False
        return time.time() > self.config.iam_token_expires_at
    
    async def ensure_valid_token(self) -> None:
        """Ensure we have a valid authentication token."""
        # For API keys, no refresh is needed
        if self.config.api_key:
            return
        
        # For IAM tokens, we could implement auto-refresh here
        if self.config.iam_token and self._is_token_expired():
            logger.warning(
                "IAM token has expired. Please refresh your token manually "
                "or use API key authentication instead."
            )

class BaseYandexService:
    """Base class for Yandex Cloud services."""
    
    def __init__(
        self,
        *,
        api_key: Optional[str] = None,
        folder_id: Optional[str] = None,
        auth_config: Optional[AuthConfig] = None,
    ):
        if auth_config is None:
            auth_config = AuthConfig(
                api_key=api_key or os.environ.get("YANDEX_API_KEY"),
                folder_id=folder_id or os.environ.get("YANDEX_FOLDER_ID"),
            )
        
        self.auth = YandexAuth(auth_config)
        self._session: Optional[aiohttp.ClientSession] = None
    
    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create HTTP session."""
        if self._session is None or self._session.closed:
            connector = aiohttp.TCPConnector(limit=100, limit_per_host=30)
            timeout = aiohttp.ClientTimeout(total=60)
            self._session = aiohttp.ClientSession(
                connector=connector,
                timeout=timeout,
                headers={"User-Agent": "livekit-yandex-plugin/0.1.0"}
            )
        return self._session
    
    async def _make_request(
        self,
        method: str,
        url: str,
        **kwargs
    ) -> aiohttp.ClientResponse:
        """Make an authenticated HTTP request."""
        await self.auth.ensure_valid_token()
        
        headers = kwargs.pop("headers", {})
        headers.update(self.auth.get_headers())
        
        session = await self._get_session()
        
        return await session.request(
            method=method,
            url=url,
            headers=headers,
            **kwargs
        )
    
    async def close(self) -> None:
        """Close the HTTP session."""
        if self._session and not self._session.closed:
            await self._session.close()
    
    async def __aenter__(self):
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()