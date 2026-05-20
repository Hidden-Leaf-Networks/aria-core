"""Tests for JWKS key management and RS256 token lifecycle."""

from __future__ import annotations

from uuid import uuid4

import pytest

try:
    from aria_core.api.jwks import KeyEntry, KeyManager, JWKSError
    from aria_core.api.auth import AuthError

    HAS_CRYPTO = True
except ImportError:
    HAS_CRYPTO = False

pytestmark = pytest.mark.skipif(not HAS_CRYPTO, reason="cryptography not installed")


class TestKeyEntry:
    def test_generate_key(self) -> None:
        entry = KeyEntry()
        assert entry.kid.startswith("aria-")
        assert not entry.revoked
        assert len(entry.private_pem) > 0
        assert len(entry.public_pem) > 0

    def test_to_jwk(self) -> None:
        entry = KeyEntry()
        jwk = entry.to_jwk()
        assert jwk["kty"] == "RSA"
        assert jwk["alg"] == "RS256"
        assert jwk["use"] == "sig"
        assert jwk["kid"] == entry.kid
        assert "n" in jwk
        assert "e" in jwk

    def test_different_keys_have_different_kids(self) -> None:
        k1 = KeyEntry()
        k2 = KeyEntry()
        assert k1.kid != k2.kid


class TestKeyManager:
    def test_generate_and_get_current(self) -> None:
        km = KeyManager()
        assert km.current_key is None

        key = km.generate_key()
        assert km.current_key is key
        assert len(km.all_keys) == 1

    def test_rotation_preserves_old_keys(self) -> None:
        km = KeyManager()
        k1 = km.generate_key()
        k2 = km.rotate()

        assert km.current_key is k2
        assert len(km.all_keys) == 2
        assert k1 in km.all_keys

    def test_revoke_key(self) -> None:
        km = KeyManager()
        k1 = km.generate_key()
        k2 = km.rotate()

        km.revoke(k1.kid)
        assert km.current_key is k2
        # Revoked key excluded from JWKS
        jwks = km.get_jwks()
        kids = [k["kid"] for k in jwks["keys"]]
        assert k1.kid not in kids
        assert k2.kid in kids

    def test_get_jwks(self) -> None:
        km = KeyManager()
        km.generate_key()
        km.generate_key()

        jwks = km.get_jwks()
        assert len(jwks["keys"]) == 2
        for key in jwks["keys"]:
            assert key["kty"] == "RSA"
            assert key["alg"] == "RS256"


class TestRS256TokenLifecycle:
    def test_create_and_decode(self) -> None:
        km = KeyManager()
        km.generate_key()
        tid = uuid4()

        token = km.create_token(
            user_id="user-1",
            tenant_id=tid,
            role="operator",
            tenant_slug="test-co",
        )

        claims = km.decode_token(token)
        assert claims["sub"] == "user-1"
        assert claims["tenant_id"] == str(tid)
        assert claims["role"] == "operator"
        assert claims["tenant_slug"] == "test-co"

    def test_token_with_rotated_key_still_valid(self) -> None:
        """Tokens signed with old key remain valid after rotation."""
        km = KeyManager()
        km.generate_key()
        tid = uuid4()

        # Sign with key 1
        token_v1 = km.create_token(user_id="user-1", tenant_id=tid)

        # Rotate to key 2
        km.rotate()
        token_v2 = km.create_token(user_id="user-2", tenant_id=tid)

        # Both tokens verify
        claims_v1 = km.decode_token(token_v1)
        assert claims_v1["sub"] == "user-1"

        claims_v2 = km.decode_token(token_v2)
        assert claims_v2["sub"] == "user-2"

    def test_expired_token_raises(self) -> None:
        km = KeyManager()
        km.generate_key()
        token = km.create_token(
            user_id="user-1",
            tenant_id=uuid4(),
            expires_in_seconds=-1,
        )
        with pytest.raises(AuthError, match="expired"):
            km.decode_token(token)

    def test_no_key_raises_on_create(self) -> None:
        km = KeyManager()
        with pytest.raises(JWKSError, match="No active signing key"):
            km.create_token(user_id="u", tenant_id=uuid4())

    def test_tampered_token_fails(self) -> None:
        km = KeyManager()
        km.generate_key()
        token = km.create_token(user_id="user-1", tenant_id=uuid4())

        # Tamper with the payload
        parts = token.split(".")
        parts[1] = parts[1][:-2] + "XX"
        tampered = ".".join(parts)

        with pytest.raises(AuthError):
            km.decode_token(tampered)

    def test_wrong_key_manager_fails(self) -> None:
        """Token from one KeyManager can't be verified by another."""
        km1 = KeyManager()
        km1.generate_key()
        token = km1.create_token(user_id="user-1", tenant_id=uuid4())

        km2 = KeyManager()
        km2.generate_key()

        with pytest.raises(AuthError):
            km2.decode_token(token)

    def test_kid_in_token_header(self) -> None:
        """Token header contains kid for key matching."""
        from jose import jwt as jose_jwt

        km = KeyManager()
        key = km.generate_key()
        token = km.create_token(user_id="u", tenant_id=uuid4())

        header = jose_jwt.get_unverified_header(token)
        assert header["kid"] == key.kid
        assert header["alg"] == "RS256"


class TestMultiKeyRotation:
    def test_three_key_rotation(self) -> None:
        """Simulate a real rotation: 3 keys, revoke oldest, all valid tokens still work."""
        km = KeyManager()
        tid = uuid4()

        k1 = km.generate_key()
        t1 = km.create_token(user_id="gen1", tenant_id=tid)

        k2 = km.rotate()
        t2 = km.create_token(user_id="gen2", tenant_id=tid)

        k3 = km.rotate()
        t3 = km.create_token(user_id="gen3", tenant_id=tid)

        # All three tokens valid
        assert km.decode_token(t1)["sub"] == "gen1"
        assert km.decode_token(t2)["sub"] == "gen2"
        assert km.decode_token(t3)["sub"] == "gen3"

        # JWKS has 3 keys
        assert len(km.get_jwks()["keys"]) == 3

        # Revoke oldest
        km.revoke(k1.kid)

        # t1 still decodes (revoked keys can still verify)
        assert km.decode_token(t1)["sub"] == "gen1"

        # JWKS now has 2 keys (revoked excluded from publication)
        assert len(km.get_jwks()["keys"]) == 2
