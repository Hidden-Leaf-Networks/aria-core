"""JWKS (JSON Web Key Set) management for RS256 JWT auth.

Provides:
- RSA key pair generation
- JWKS endpoint serving public keys
- Key rotation with grace period (old keys remain valid)
- Token creation with kid (Key ID) headers

Usage:
    from aria_core.api.jwks import KeyManager

    km = KeyManager()
    km.generate_key()
    token = km.create_token(user_id="u1", tenant_id=tid, role="operator")
    claims = km.decode_token(token)  # verifies with matching public key
"""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from typing import Any
from uuid import UUID, uuid4

try:
    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.asymmetric import rsa
    from jose import jwt, JWTError, ExpiredSignatureError

    HAS_CRYPTO = True
except ImportError:
    HAS_CRYPTO = False


class JWKSError(Exception):
    pass


class KeyEntry:
    """A single RSA key pair with metadata."""

    def __init__(
        self,
        kid: str | None = None,
        key_size: int = 2048,
        private_key: Any = None,
    ) -> None:
        if not HAS_CRYPTO:
            raise JWKSError("cryptography + python-jose required: pip install 'aria-core[api]'")

        self.kid = kid or f"aria-{uuid4().hex[:12]}"
        self.created_at = datetime.now(timezone.utc)
        self.revoked = False

        if private_key:
            self._private_key = private_key
        else:
            self._private_key = rsa.generate_private_key(
                public_exponent=65537,
                key_size=key_size,
            )

    @property
    def public_key(self) -> Any:
        return self._private_key.public_key()

    @property
    def private_pem(self) -> bytes:
        return self._private_key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=serialization.NoEncryption(),
        )

    @property
    def public_pem(self) -> bytes:
        return self.public_key.public_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo,
        )

    def to_jwk(self) -> dict[str, Any]:
        """Export public key as JWK (JSON Web Key) format."""
        from cryptography.hazmat.primitives.asymmetric.rsa import RSAPublicNumbers
        import base64

        pub_numbers: RSAPublicNumbers = self.public_key.public_numbers()

        def _int_to_base64url(n: int) -> str:
            byte_length = (n.bit_length() + 7) // 8
            data = n.to_bytes(byte_length, byteorder="big")
            return base64.urlsafe_b64encode(data).rstrip(b"=").decode("ascii")

        return {
            "kty": "RSA",
            "kid": self.kid,
            "use": "sig",
            "alg": "RS256",
            "n": _int_to_base64url(pub_numbers.n),
            "e": _int_to_base64url(pub_numbers.e),
        }


class KeyManager:
    """Manages RSA key pairs for RS256 JWT signing and verification.

    Supports key rotation: new keys are generated, old keys remain
    valid for verification during a grace period.
    """

    def __init__(self, issuer: str = "aria-core", audience: str = "aria-core-api") -> None:
        self.issuer = issuer
        self.audience = audience
        self._keys: list[KeyEntry] = []

    @property
    def current_key(self) -> KeyEntry | None:
        """The most recent non-revoked key (used for signing)."""
        active = [k for k in self._keys if not k.revoked]
        return active[-1] if active else None

    @property
    def all_keys(self) -> list[KeyEntry]:
        """All keys including revoked (for verification grace period)."""
        return list(self._keys)

    def generate_key(self, key_size: int = 2048) -> KeyEntry:
        """Generate a new RSA key pair and make it the active signing key."""
        entry = KeyEntry(key_size=key_size)
        self._keys.append(entry)
        return entry

    def rotate(self, key_size: int = 2048) -> KeyEntry:
        """Generate a new key. Old keys remain valid for verification."""
        return self.generate_key(key_size)

    def revoke(self, kid: str) -> bool:
        """Revoke a key by kid. Revoked keys are excluded from JWKS but
        can still verify tokens during grace period if needed."""
        for key in self._keys:
            if key.kid == kid:
                key.revoked = True
                return True
        return False

    def get_jwks(self) -> dict[str, Any]:
        """Get the JWKS (public keys only, non-revoked)."""
        return {
            "keys": [k.to_jwk() for k in self._keys if not k.revoked],
        }

    def create_token(
        self,
        user_id: str,
        tenant_id: UUID,
        role: str = "operator",
        tenant_slug: str | None = None,
        expires_in_seconds: int = 3600,
    ) -> str:
        """Create a signed JWT using the current key."""
        key = self.current_key
        if not key:
            raise JWKSError("No active signing key — call generate_key() first")

        now = datetime.now(timezone.utc)
        payload = {
            "sub": user_id,
            "tenant_id": str(tenant_id),
            "role": role,
            "iss": self.issuer,
            "aud": self.audience,
            "iat": int(now.timestamp()),
            "exp": int(now.timestamp()) + expires_in_seconds,
        }
        if tenant_slug:
            payload["tenant_slug"] = tenant_slug

        return jwt.encode(
            payload,
            key.private_pem,
            algorithm="RS256",
            headers={"kid": key.kid},
        )

    def decode_token(self, token: str) -> dict[str, Any]:
        """Decode and verify a JWT, matching kid to the correct public key."""
        from aria_core.api.auth import AuthError

        try:
            unverified_header = jwt.get_unverified_header(token)
        except JWTError:
            raise AuthError("Invalid token header")

        kid = unverified_header.get("kid")

        # Find matching key
        matching_key = None
        for key in self._keys:
            if key.kid == kid:
                matching_key = key
                break

        if not matching_key:
            # Try all keys (fallback for tokens without kid)
            for key in reversed(self._keys):
                try:
                    return jwt.decode(
                        token,
                        key.public_pem,
                        algorithms=["RS256"],
                        issuer=self.issuer,
                        audience=self.audience,
                    )
                except JWTError:
                    continue
            raise AuthError("No matching key found for token")

        try:
            return jwt.decode(
                token,
                matching_key.public_pem,
                algorithms=["RS256"],
                issuer=self.issuer,
                audience=self.audience,
            )
        except ExpiredSignatureError:
            raise AuthError("Token has expired")
        except JWTError as e:
            raise AuthError(f"Invalid token: {e}")
