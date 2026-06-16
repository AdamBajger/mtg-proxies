from __future__ import annotations

from pathlib import Path
from typing import Literal

from tqdm import tqdm

from mtgproxies import scryfall
from mtgproxies.decklists.decklist import Card, Decklist


def fetch_scans_scryfall(decklist: Decklist, cache_dir: Path,
                         faces: Literal["all", "front", "back"] = "all") -> list[Path]:
    """Search Scryfall for scans of a decklist.

    Args:
        decklist: The decklist to fetch scans for
        faces: Which faces to fetch ("all", "front", "back")
        cache_dir: Directory to cache the images

    Returns:
        List: List of image files
    """
    return [
        scan
        for card in tqdm(decklist.cards, desc="Fetching artwork")
        for i, image_uri in enumerate(card.image_uris)
        for scan in [scryfall.get_image(image_uri["png"], silent=True, cache_dir=cache_dir)] * card.count
        if faces == "all" or (faces == "front" and i == 0) or (faces == "back" and i > 0)
    ]


class CardSlot:
    """A single physical card's image pairing for duplex printing.

    For single-faced cards, ``back`` is None (generic card back used).
    For double-faced cards (transform, MDFC, etc.), ``back`` holds the
    actual back face image path.
    """

    def __init__(self, front: Path, card: Card, back: Path | None = None):
        self.front = front
        self.card = card
        self.back = back

    @property
    def is_double_faced(self) -> bool:
        return self.back is not None


def prepare_card_slots(decklist: Decklist, cache_dir: Path) -> list[CardSlot]:
    """Fetch card images and pair fronts with their potential backs.

    Downloads all card face images via Scryfall, then groups them into
    :class:`CardSlot` objects, one per physical card copy.  Single-faced
    cards get ``back=None``; double-faced cards (transform, MDFC, etc.)
    pair the actual back face image.

    Args:
        decklist: The decklist to fetch scans for
        cache_dir: Directory to cache the images

    Returns:
        List of CardSlot objects, one per card copy.
    """
    slots: list[CardSlot] = []
    for card in tqdm(decklist.cards, desc="Fetching artwork"):
        uris = list(card.image_uris)
        for _ in range(card.count):
            front_path = scryfall.get_image(uris[0]["png"], silent=True, cache_dir=cache_dir)
            back_path = scryfall.get_image(uris[1]["png"], silent=True, cache_dir=cache_dir) if len(uris) > 1 else None
            slots.append(CardSlot(front=front_path, card=card, back=back_path))
    return slots
