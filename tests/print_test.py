import math
from pathlib import Path

import pytest


@pytest.fixture(scope="module")
def example_images() -> list[str]:
    from mtgproxies import fetch_scans_scryfall
    from mtgproxies.decklists import parse_decklist

    decklist, _, _ = parse_decklist(Path(__file__).parent.parent / "examples/decklist.txt")
    images = fetch_scans_scryfall(decklist)

    return images


def test_example_images(example_images: list[str]):
    assert len(example_images) == 7


def test_print_cards_fpdf(example_images: list[str], tmp_path: Path):
    from mtgproxies import print_cards_fpdf

    out_file = tmp_path / "decklist.pdf"
    print_cards_fpdf(example_images, out_file)

    assert out_file.is_file()


def test_print_cards_matplotlib_pdf(example_images: list[str], tmp_path: Path):
    from mtgproxies import print_cards_matplotlib

    out_file = tmp_path / "decklist.pdf"
    print_cards_matplotlib(example_images, out_file)

    assert out_file.is_file()


@pytest.mark.skip(reason="for some reason this fails on github actions, but works locally.")
def test_print_cards_matplotlib_png(example_images: list[str], tmp_path: Path):
    from mtgproxies import print_cards_matplotlib

    out_file = tmp_path / "decklist.png"
    print_cards_matplotlib(example_images, out_file)

    assert (tmp_path / "decklist_000.png").is_file()


def test_dimension_units_coverage():
    from mtgproxies.dimensions import Units, PAPER_SIZE

    for unit in Units.__args__:
        for spec in PAPER_SIZE:
            assert unit in PAPER_SIZE[spec]

def test_units_conversion_to_mm():
    from mtgproxies.dimensions import UNITS_TO_MM

    assert math.isclose(6 * UNITS_TO_MM["in"], 152.4, rel_tol=1e-3)
    assert math.isclose(6 * UNITS_TO_MM["cm"], 60, rel_tol=1e-3)
    assert math.isclose(6 * UNITS_TO_MM["mm"], 6, rel_tol=1e-3)

    assert math.isclose(152.4 / UNITS_TO_MM["in"], 6, rel_tol=1e-3)
    assert math.isclose(60 / UNITS_TO_MM["cm"], 6, rel_tol=1e-3)
    assert math.isclose(6 / UNITS_TO_MM["mm"], 6, rel_tol=1e-3)

