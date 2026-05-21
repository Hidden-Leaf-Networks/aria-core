"""Tests for agent marketplace."""

from __future__ import annotations

from uuid import uuid4

import pytest

from aria_core.archetypes.marketplace import Marketplace, MarketplaceListing
from aria_core.archetypes.models import Archetype, ArchetypeCategory
from aria_core.archetypes.registry import ArchetypeRegistry


@pytest.fixture
async def marketplace() -> Marketplace:
    registry = ArchetypeRegistry()
    mp = Marketplace(registry)
    return mp


@pytest.fixture
def tenant_id() -> uuid4:
    return uuid4()


async def _seed_and_publish(mp: Marketplace, tid: uuid4) -> MarketplaceListing:
    await mp._registry.seed_defaults(tid)
    archetypes = await mp._registry.list(tid)
    return await mp.publish(tid, archetypes[0].id, publisher_name="Test Publisher")


class TestPublish:
    async def test_publish_archetype(self, marketplace: Marketplace, tenant_id: uuid4) -> None:
        listing = await _seed_and_publish(marketplace, tenant_id)
        assert listing.publisher_id == tenant_id
        assert listing.downloads == 0
        assert marketplace.listing_count == 1

    async def test_publish_nonexistent_raises(self, marketplace: Marketplace) -> None:
        with pytest.raises(ValueError, match="not found"):
            await marketplace.publish(uuid4(), uuid4())


class TestBrowse:
    async def test_browse_all(self, marketplace: Marketplace, tenant_id: uuid4) -> None:
        await _seed_and_publish(marketplace, tenant_id)
        listings = await marketplace.browse()
        assert len(listings) == 1

    async def test_browse_by_category(self, marketplace: Marketplace, tenant_id: uuid4) -> None:
        await mp._registry.seed_defaults(tenant_id) if (mp := marketplace) else None
        archetypes = await marketplace._registry.list(tenant_id)

        for a in archetypes[:3]:
            await marketplace.publish(tenant_id, a.id)

        research = await marketplace.browse(category="research")
        for l in research:
            assert l.archetype.category == ArchetypeCategory.RESEARCH

    async def test_browse_search(self, marketplace: Marketplace, tenant_id: uuid4) -> None:
        listing = await _seed_and_publish(marketplace, tenant_id)
        name = listing.archetype.name.split()[0].lower()
        results = await marketplace.browse(search=name)
        assert len(results) >= 1

    async def test_browse_sort_by_rating(self, marketplace: Marketplace, tenant_id: uuid4) -> None:
        await marketplace._registry.seed_defaults(tenant_id)
        archetypes = await marketplace._registry.list(tenant_id)
        l1 = await marketplace.publish(tenant_id, archetypes[0].id)
        l2 = await marketplace.publish(tenant_id, archetypes[1].id)

        await marketplace.rate(l2.id, "user-1", tenant_id, 5)
        results = await marketplace.browse(sort_by="rating")
        assert results[0].rating >= results[-1].rating


class TestInstall:
    async def test_install_listing(self, marketplace: Marketplace, tenant_id: uuid4) -> None:
        listing = await _seed_and_publish(marketplace, tenant_id)
        other_tenant = uuid4()

        installed = await marketplace.install(other_tenant, listing.id)
        assert installed.tenant_id == other_tenant
        assert installed.is_builtin is False
        assert "marketplace_listing_id" in installed.metadata

        # Download count incremented
        updated = await marketplace.get_listing(listing.id)
        assert updated is not None
        assert updated.downloads == 1

    async def test_install_nonexistent_raises(self, marketplace: Marketplace) -> None:
        with pytest.raises(ValueError, match="not found"):
            await marketplace.install(uuid4(), uuid4())


class TestRating:
    async def test_rate_listing(self, marketplace: Marketplace, tenant_id: uuid4) -> None:
        listing = await _seed_and_publish(marketplace, tenant_id)

        review = await marketplace.rate(listing.id, "user-1", tenant_id, 4, "Great agent!")
        assert review.rating == 4

        updated = await marketplace.get_listing(listing.id)
        assert updated is not None
        assert updated.rating == 4.0
        assert updated.rating_count == 1

    async def test_multiple_ratings_average(self, marketplace: Marketplace, tenant_id: uuid4) -> None:
        listing = await _seed_and_publish(marketplace, tenant_id)

        await marketplace.rate(listing.id, "user-1", tenant_id, 5)
        await marketplace.rate(listing.id, "user-2", tenant_id, 3)

        updated = await marketplace.get_listing(listing.id)
        assert updated is not None
        assert updated.rating == 4.0
        assert updated.rating_count == 2

    async def test_get_reviews(self, marketplace: Marketplace, tenant_id: uuid4) -> None:
        listing = await _seed_and_publish(marketplace, tenant_id)
        await marketplace.rate(listing.id, "u1", tenant_id, 5, "Excellent")
        await marketplace.rate(listing.id, "u2", tenant_id, 3, "OK")

        reviews = await marketplace.get_reviews(listing.id)
        assert len(reviews) == 2


class TestUnlist:
    async def test_unlist(self, marketplace: Marketplace, tenant_id: uuid4) -> None:
        listing = await _seed_and_publish(marketplace, tenant_id)
        assert await marketplace.unlist(listing.id) is True
        assert marketplace.listing_count == 0

    async def test_unlist_nonexistent(self, marketplace: Marketplace) -> None:
        assert await marketplace.unlist(uuid4()) is False
