"""Agent Marketplace — publish, discover, install, and rate archetypes.

Community-driven marketplace built on the archetype registry.
Archetypes can be published (shared publicly), installed (copied to tenant),
and rated by users.

Usage:
    from aria_core.archetypes.marketplace import Marketplace

    mp = Marketplace()
    await mp.publish(tenant_id, archetype_id)
    listings = await mp.browse(category="research")
    await mp.install(tenant_id, listing_id)
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any
from uuid import UUID, uuid4

from pydantic import Field

from aria_core.archetypes.models import Archetype, ArchetypeCategory
from aria_core.archetypes.registry import ArchetypeRegistry
from aria_core.runtime.models import BaseModel


class MarketplaceListing(BaseModel):
    """A published archetype in the marketplace."""

    id: UUID = Field(default_factory=uuid4)
    archetype: Archetype
    publisher_id: UUID
    publisher_name: str = ""
    downloads: int = 0
    rating: float = 0.0
    rating_count: int = 0
    featured: bool = False
    tags: list[str] = Field(default_factory=list)
    published_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class MarketplaceReview(BaseModel):
    """A user review of a marketplace listing."""

    id: UUID = Field(default_factory=uuid4)
    listing_id: UUID
    user_id: str
    tenant_id: UUID
    rating: int = Field(ge=1, le=5)
    comment: str = ""
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class Marketplace:
    """Agent archetype marketplace — publish, discover, install, rate."""

    def __init__(self, registry: ArchetypeRegistry | None = None) -> None:
        self._registry = registry or ArchetypeRegistry()
        self._listings: dict[UUID, MarketplaceListing] = {}
        self._reviews: dict[UUID, list[MarketplaceReview]] = {}

    async def publish(
        self,
        tenant_id: UUID,
        archetype_id: UUID,
        publisher_name: str = "",
    ) -> MarketplaceListing:
        """Publish an archetype to the marketplace."""
        archetype = await self._registry.get(tenant_id, archetype_id)
        if not archetype:
            raise ValueError(f"Archetype {archetype_id} not found")

        listing = MarketplaceListing(
            archetype=archetype,
            publisher_id=tenant_id,
            publisher_name=publisher_name,
            tags=list(archetype.tags),
        )
        self._listings[listing.id] = listing
        return listing

    async def browse(
        self,
        category: str | None = None,
        search: str | None = None,
        sort_by: str = "downloads",
        limit: int = 50,
        offset: int = 0,
    ) -> list[MarketplaceListing]:
        """Browse marketplace listings."""
        listings = list(self._listings.values())

        if category:
            listings = [l for l in listings if l.archetype.category.value == category]

        if search:
            q = search.lower()
            listings = [
                l for l in listings
                if q in l.archetype.name.lower()
                or q in l.archetype.description.lower()
                or any(q in t.lower() for t in l.tags)
            ]

        if sort_by == "downloads":
            listings.sort(key=lambda l: l.downloads, reverse=True)
        elif sort_by == "rating":
            listings.sort(key=lambda l: l.rating, reverse=True)
        elif sort_by == "newest":
            listings.sort(key=lambda l: l.published_at, reverse=True)

        return listings[offset : offset + limit]

    async def get_listing(self, listing_id: UUID) -> MarketplaceListing | None:
        return self._listings.get(listing_id)

    async def install(
        self, tenant_id: UUID, listing_id: UUID
    ) -> Archetype:
        """Install a marketplace archetype to a tenant."""
        listing = self._listings.get(listing_id)
        if not listing:
            raise ValueError(f"Listing {listing_id} not found")

        # Copy archetype to tenant
        installed = listing.archetype.model_copy(update={
            "id": uuid4(),
            "tenant_id": tenant_id,
            "is_builtin": False,
            "metadata": {
                **listing.archetype.metadata,
                "marketplace_listing_id": str(listing_id),
                "installed_from": listing.publisher_name,
            },
        })
        await self._registry.save(tenant_id, installed)

        # Increment download count
        listing = listing.model_copy(update={"downloads": listing.downloads + 1})
        self._listings[listing_id] = listing

        return installed

    async def rate(
        self,
        listing_id: UUID,
        user_id: str,
        tenant_id: UUID,
        rating: int,
        comment: str = "",
    ) -> MarketplaceReview:
        """Rate a marketplace listing (1-5 stars)."""
        listing = self._listings.get(listing_id)
        if not listing:
            raise ValueError(f"Listing {listing_id} not found")

        review = MarketplaceReview(
            listing_id=listing_id,
            user_id=user_id,
            tenant_id=tenant_id,
            rating=max(1, min(5, rating)),
            comment=comment,
        )

        if listing_id not in self._reviews:
            self._reviews[listing_id] = []
        self._reviews[listing_id].append(review)

        # Recalculate average rating
        reviews = self._reviews[listing_id]
        avg = sum(r.rating for r in reviews) / len(reviews)
        listing = listing.model_copy(update={
            "rating": round(avg, 1),
            "rating_count": len(reviews),
        })
        self._listings[listing_id] = listing

        return review

    async def get_reviews(
        self, listing_id: UUID, limit: int = 50
    ) -> list[MarketplaceReview]:
        reviews = self._reviews.get(listing_id, [])
        return sorted(reviews, key=lambda r: r.created_at, reverse=True)[:limit]

    async def unlist(self, listing_id: UUID) -> bool:
        if listing_id in self._listings:
            del self._listings[listing_id]
            return True
        return False

    @property
    def listing_count(self) -> int:
        return len(self._listings)
