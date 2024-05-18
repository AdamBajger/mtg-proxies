# standard imports
from __future__ import annotations

import abc
from logging import getLogger
from typing import TYPE_CHECKING

import numpy as np

# third-party imports
import PIL
from tqdm import tqdm

# local imports
from mtgproxies.dimensions import Units, get_pixels_from_size_and_ppsu, get_ppsu_from_size_and_pixels
from mtgproxies.print_cards import fill_corners


if TYPE_CHECKING:
    from collections.abc import Generator
    from pathlib import Path

    from nptyping import Float, NDArray
    from PIL.Image import Image


logger = getLogger(__name__)
Bbox = tuple[float, float, float, float]  # (x, y, width, height)
Lcoords = tuple[float, float, float, float]  # (x0, y0, x1, y1)


class CardAssembler(abc.ABC):
    """Base class for assembling cards into sheets."""

    def __init__(
        self,
        paper_size: NDArray[2, Float],
        card_size: NDArray[2, Float],
        border_crop: float = 0,
        crop_marks_thickness: float = 0,
        cut_spacing_thickness: float = 0,
        filled_corners: bool = False,
        background_color: tuple[int, int, int] | None = None,
        page_safe_margin: float = 0,
        units: Units = "mm",
    ):
        """Initialize the CardAssembler.

        Args:
            paper_size: Size of the paper in the specified units.
            card_size: Size of a card in the specified units.
            border_crop: How many units to crop from the border of each card.
            crop_marks_thickness: Thickness of crop marks in the specified units. Use 0 to disable crop marks.
            cut_spacing_thickness: Thickness of cut lines in the specified units. Use 0 to disable cut lines.
            filled_corners: Whether to fill in the corners of the cards.
            background_color: Background color of the paper.
            page_safe_margin: How much to leave as a margin on the paper.
            units: Units to use for the sizes.
        """
        # self.mm_coeff = UNITS_TO_MM[units]
        self.units = units
        self.paper_size = paper_size
        self.paper_safe_margin = page_safe_margin
        self.card_size = card_size
        self.border_crop = border_crop
        self.crop_marks_thickness = crop_marks_thickness
        self.cut_spacing_thickness = cut_spacing_thickness
        self.background_color = background_color
        self.filled_corners = filled_corners

        # precompute some values
        self.card_bbox_size = self.card_size - (self.border_crop * 2)
        self.safe_printable_area = self.paper_size - (self.paper_safe_margin * 2)
        self.grid_dims = (
            (self.safe_printable_area + self.cut_spacing_thickness)
            // (self.card_bbox_size + self.cut_spacing_thickness)
        ).astype(np.int32)
        self.rows, self.cols = self.grid_dims
        self.grid_bbox_size = self.card_bbox_size * self.grid_dims + self.cut_spacing_thickness * (self.grid_dims - 1)
        self.offset = (self.safe_printable_area - self.grid_bbox_size) / 2

    @abc.abstractmethod
    def assemble(self, card_image_filepaths: list[Path], output_filepath: Path):
        ...

    def process_card_image(self, card_image_filepath: Path) -> Image:
        """Process an image for assembly.

        Loads the image, fills in corners, crops the borders, etc.

        Args:
            card_image_filepath: Image file to process.
        """
        img = PIL.Image.open(card_image_filepath).copy()
        # fill corners
        if self.filled_corners:
            img = fill_corners(img)
        # crop the cards
        ppsu = get_ppsu_from_size_and_pixels(pixel_values=img.size, size=self.card_size)
        crop_px = int(get_pixels_from_size_and_ppsu(ppsu=ppsu, size=self.border_crop))

        img = img.crop(box=(crop_px, crop_px, img.width - crop_px, img.height - crop_px))
        return img

    def get_page_generators(
        self,
        card_image_filepaths: list[str | Path],
    ) -> Generator[Generator[tuple[Bbox, NDArray]]]:
        """This method is a generator of generators of bounding boxes for each card on a page and the page indices.

        The method can be iterated over to get the bbox iterators for each page and its index.

        Yields:
            tuple[Generator[Bbox], int]: Generator of bounding boxes and page index.
        """
        per_sheet = self.rows * self.cols
        remaining = card_image_filepaths
        while remaining:
            if len(remaining) >= per_sheet:
                for_sheet = remaining[:per_sheet]
                remaining = remaining[per_sheet:]
            else:
                for_sheet = remaining
                remaining = []

            yield self.get_bbox_generator(for_sheet)

    def get_bbox_generator(self, cards_on_page: list[Path]) -> Generator[tuple[Bbox, Image]]:
        """This method is a generator of bounding boxes for each card on a page.

        The method takes a list of card image filepaths and yields the bounding boxes for each of those cards.
        The bounding boxes are tiling a precalculated grid on the page.

        Args:
            cards_on_page: Number of cards on the page.

        Yields:
            Bbox: Bounding box for each card. The format is (x, y, width, height).
        """
        for i, card_image_filepath in enumerate(cards_on_page):
            card_pos = np.array([i % self.cols, i // self.cols])

            cut_spacing_offset = self.cut_spacing_thickness * card_pos
            preceding_cards_offset = self.card_bbox_size * card_pos
            card_offset = cut_spacing_offset + preceding_cards_offset + self.paper_safe_margin + self.offset

            image = self.process_card_image(card_image_filepath)

            yield (*card_offset, *self.card_bbox_size), image

    def get_line_generator(self) -> Generator[Lcoords]:
        """This method is a generator of line coordinates for crop marks.

        The crop marks are lining the edges of each card to help with aligning both printed sides together.

        Yields:
            Bbox: Coordinate points for each cut line. The format is (x0, y0, x1, y1).
        """
        # Horizontal lines
        for i in range(self.rows):
            y_top = self.paper_safe_margin + self.offset[1] + i * (self.card_bbox_size[1] + self.cut_spacing_thickness)
            y_bottom = y_top + self.card_bbox_size[1]
            yield 0, y_top, self.paper_size[0], y_top
            yield 0, y_bottom, self.paper_size[0], y_bottom

        # Vertical lines
        for i in range(self.cols):
            x_left = self.paper_safe_margin + self.offset[0] + i * (self.card_bbox_size[0] + self.cut_spacing_thickness)
            x_right = x_left + self.card_bbox_size[0]
            yield x_left, self.paper_size[1], x_left, 0
            yield x_right, self.paper_size[1], x_right, 0

    def prepare_routine(self, card_image_filepaths, output_filepath):
        total_cards = len(card_image_filepaths)
        pages = np.ceil(total_cards / (self.rows * self.cols)).astype(int)
        tqdm_ = tqdm(total=total_cards, desc="Plotting cards")
        logger.info(f"Will print {total_cards} cards in {pages} pages in a {self.rows}x{self.cols} grid.")
        # Ensure parent directory exists for the output file
        output_filepath.parent.mkdir(parents=True, exist_ok=True)
        return pages, tqdm_
