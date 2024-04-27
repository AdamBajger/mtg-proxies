from __future__ import annotations

import abc
from pathlib import Path
import logging
from logging import getLogger
from typing import Any, Literal, Generator, Iterable

import PIL
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from nptyping import NDArray, Float, Int, Shape
import numpy as np

from fpdf import FPDF
from PIL.Image import Image

from tqdm import tqdm
from mtgproxies.plotting import SplitPages
from mtgproxies.dimensions import get_pixels_from_size_and_ppsu, MTG_CARD_INCHES, get_ppsu_from_size_and_pixels, \
    MTG_CARD_MM, UNITS_TO_MM, Units, PAPER_SIZE

image_size = np.array([745, 1040])

logger = getLogger(__name__)


def _occupied_space(cardsize, pos, border_crop: int, closed: bool = False):

    old_result = cardsize * (pos * image_size - np.clip(2 * pos - 1 - closed, 0, None) * border_crop) / image_size

    new_result = occupied_space_new(cardsize, pos, border_crop=border_crop, gap_size=0)
    logger.info(f"Old result: {old_result}, New result: {new_result},\nDiff: {old_result - new_result}")
    return old_result


def occupied_space_new(card_size: NDArray[2, Float], row_cols: NDArray[2, Int], border_crop: float, gap_size: float = 0) -> NDArray[2, Float]:
    """Calculate the space occupied by cards on a sheet.

    Args:
        card_size: Size of a card (in inches).
        row_cols: Number of rows and columns of cards.
        border_crop: How much to crop from the border of each card (in inches)
        gap_size: Size of the gap between cards (in inches)

    Returns:
        Size of the occupied space (in inches).
    """
    cropped_card_size = card_size - 2 * border_crop
    total_gap_area = (row_cols - 1) * gap_size
    return row_cols * cropped_card_size + total_gap_area


def print_cards_matplotlib(
    images: list[str | Path],
    filepath: str | Path,
    papersize=PAPER_SIZE["A4"]['in'],
    cardsize=MTG_CARD_INCHES,
    border_crop: int = 14,
    interpolation: str | None = "lanczos",
    dpi: int = 600,
    background_color=None,
):
    """Print a list of cards to a pdf file.

    Args:
        images: List of image files
        filepath: Name of the pdf file
        papersize: Size of the paper in inches. Defaults to A4.
        cardsize: Size of a card in inches.
        border_crop: How many pixel to crop from the border of each card.
        interpolation: Interpolation method for resizing images.
        dpi: Dots per inch for the output file.
        background_color: Background color of the paper.
    """
    # Cards per figure
    dims = np.floor(papersize / cardsize).astype(int)
    rows, cols = dims
    if rows == 0 or cols == 0:
        raise ValueError(f"Paper size too small: {papersize}")
    # calculate the offset to center the cards on the paper
    offset = (papersize - _occupied_space(cardsize, dims, border_crop, closed=True)) / 2


    # Ensure directory exists
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    with SplitPages(filepath) as saver, tqdm(total=len(images), desc="Plotting cards") as pbar:
        while len(images) > 0:
            # create a new figure of paper size and an axes that covers the whole figure
            fig = plt.figure(figsize=papersize)  # type: ignore
            ax = fig.add_axes(rect=(0, 0, 1, 1))  # 0,0 offset fractions, 1,1 size fractions

            #  Background color
            if background_color is not None:
                plt.gca().add_patch(Rectangle((0, 0), 1, 1, color=background_color, zorder=-1000))

            for y in range(cols):
                for x in range(rows):
                    if len(images) > 0:
                        img = plt.imread(images.pop(0))
                        img_dpi = get_ppsu_from_size_and_pixels(pixel_values=img.shape[:2], size=cardsize)

                        # Crop left and top if not on border of sheet
                        # left = border_crop if x > 0 else 0
                        # top = border_crop if y > 0 else 0
                        border_crop_pixels = get_pixels_from_size_and_ppsu(ppsu=img_dpi, size=border_crop)

                        img = img[border_crop_pixels:, border_crop_pixels:]

                        # Compute extent (the box that will be filled by the image) as [left, right, bottom, top]
                        occupied_space_before = _occupied_space(cardsize, np.array([x, y]), border_crop)
                        lower = (offset + occupied_space_before) / papersize
                        upper = (
                            offset
                            + occupied_space_before
                            + cardsize * (image_size - border_crop_pixels) / image_size
                        ) / papersize
                        # flip y-axis
                        extent = (float(lower[0]), float(upper[0]), 1 - float(upper[1]), 1 - float(lower[1]))

                        plt.imshow(
                            img,
                            extent=extent,
                            # aspect=papersize[1] / papersize[0],
                            interpolation=interpolation,
                        )
                        pbar.update(1)

            plt.xlim(0, 1)
            plt.ylim(0, 1)

            # Hide all axis ticks and labels
            ax.axis("off")

            saver.savefig(dpi=dpi)
            plt.close()


def print_cards_fpdf(
    images: list[str | Path],
    filepath: str | Path,
    papersize=PAPER_SIZE["A4"]['mm'],
    cardsize=MTG_CARD_MM,
    border_crop: int = 14,
    background_color: tuple[int, int, int] = None,
    cropmarks: bool = True,
) -> None:
    """Print a list of cards to a pdf file.

    Args:
        images: List of image files
        filepath: Name of the pdf file
        papersize: Size of the paper in inches. Defaults to A4.
        cardsize: Size of a card in inches.
        border_crop: How many pixel to crop from the border of each card.
    """


    # Cards per sheet
    dims = np.floor(papersize / cardsize).astype(int)
    rows, cols = dims
    if rows == 0 or cols == 0:
        raise ValueError(f"Paper size too small: {papersize}")
    cards_per_sheet = np.prod(dims)
    offset = (papersize - _occupied_space(cardsize, dims, border_crop, closed=True)) / 2

    # Ensure directory exists
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    # Initialize PDF
    pdf = FPDF(orientation="P", unit="mm", format=papersize)  # type: ignore

    for i, image in enumerate(tqdm(images, desc="Plotting cards")):
        if i % cards_per_sheet == 0:  # Starting a new sheet
            pdf.add_page()
            if background_color is not None:
                pdf.set_fill_color(*background_color)
                pdf.rect(0, 0, float(papersize[0]), float(papersize[1]), "F")

        # Compute card position on sheet
        x = (i % cards_per_sheet) % rows
        y = (i % cards_per_sheet) // rows

        # Crop left and top if not on border of sheet
        left = border_crop if x > 0 else 0
        top = border_crop if y > 0 else 0

        if left == 0 and top == 0:
            cropped_image = image
        else:
            path = Path(image)
            cropped_image = str(path.parent / (path.stem + f"_{left}_{top}" + path.suffix))
            if not Path(cropped_image).is_file():
                # Crop image
                plt.imsave(cropped_image, plt.imread(image)[top:, left:])

        # Compute extent (the box that will be filled by the image) as [x, y, width, height]
        lower = offset + _occupied_space(cardsize, np.array([x, y]), border_crop)
        size = cardsize * (image_size - [left, top]) / image_size

        # Plot image
        pdf.image(cropped_image, x=lower[0], y=lower[1], w=size[0], h=size[1])

        if cropmarks and ((i + 1) % cards_per_sheet == 0 or i + 1 == len(images)):
            # If this was the last card on a page, add crop marks
            pdf.set_line_width(0.05)
            pdf.set_draw_color(255, 255, 255)
            a = cardsize * (image_size - 2 * border_crop) / image_size
            b = papersize - dims * a
            for x in range(rows + 1):
                for y in range(cols + 1):
                    mark = b / 2 + a * [x, y]
                    pdf.line(mark[0] - 0.5, mark[1], mark[0] + 0.5, mark[1])
                    pdf.line(mark[0], mark[1] - 0.5, mark[0], mark[1] + 0.5)

    tqdm.write(f"Writing to {filepath}")
    pdf.output(str(filepath))


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
            cut_lines_thickness: float = 0,
            fill_corners: bool = False,
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
            cut_lines_thickness: Thickness of cut lines in the specified units. Use 0 to disable cut lines.
            fill_corners: Whether to fill in the corners of the cards.
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
        self.cut_lines_thickness = cut_lines_thickness
        self.background_color = background_color
        self.fill_corners_ = fill_corners

        # precompute some values
        self.card_bbox_size = self.card_size - (self.border_crop * 2)
        self.safe_printable_area = self.paper_size - (self.paper_safe_margin * 2)
        self.grid_dims = ((self.safe_printable_area + self.cut_lines_thickness) //
                          (self.card_bbox_size + self.cut_lines_thickness)).astype(int)
        self.rows, self.cols = self.grid_dims
        self.grid_bbox_size = self.card_bbox_size * self.grid_dims + self.cut_lines_thickness * (self.grid_dims - 1)
        self.offset = (self.safe_printable_area - self.grid_bbox_size) / 2

    @abc.abstractmethod
    def assemble(self, card_image_filepaths: list[str | Path], output_filepath: str | Path):
        ...

    def process_card_image(self, card_image_filepath: str | Path) -> Image:
        """Process an image for assembly.

        Loads the image, fills in corners, crops the borders, etc.

        Args:
            card_image_filepath: Image file to process.
        """
        img = np.asarray(PIL.Image.open(card_image_filepath))
        # fill corners
        if self.fill_corners_:
            img = self.fill_corners(img)
        # crop the cards
        ppsu = get_ppsu_from_size_and_pixels(pixel_values=img.shape[:2], size=self.card_size)
        crop_px = get_pixels_from_size_and_ppsu(ppsu=ppsu, size=self.border_crop)
        img = img[crop_px:, crop_px:]
        return img

    def fill_corners(self, img: NDArray[Shape["2"], Float]) -> NDArray[Shape["2"], Float]:
        """Fill the corners of the card with the closest pixels around the corners to match the border color."""
        logger.warning("Filling corners not implemented, returning original image.")
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

        Args:
            cards_on_page: Number of cards on the page.

        Yields:
            Bbox: Bounding box for each card.
        """
        for i, card_image_filepath in enumerate(cards_on_page):
            card_pos = np.array([i % self.rows, i // self.rows])

            cut_lines_offset = self.cut_lines_thickness * card_pos
            preceding_cards_offset = self.card_bbox_size * card_pos
            card_offset = cut_lines_offset + preceding_cards_offset + self.paper_safe_margin + self.offset
            image = self.process_card_image(card_image_filepath)

            yield (*card_offset, *self.card_bbox_size), image

    def get_line_generator(self) -> Generator[Lcoords]:
        """This method is a generator of bounding boxes for each cut line on a page.

        Yields:
            Bbox: Bounding box for each cut line.
        """
        # Horizontal lines
        for i in range(self.rows):
            y_left = self.paper_safe_margin + self.offset[1] + i * (self.card_bbox_size + self.cut_lines_thickness)
            y_right = y_left + self.card_bbox_size
            yield 0, y_left, self.paper_size[0], y_left
            yield 0, y_right, self.paper_size[0], y_right

        for i in range(self.cols):
            x_top = self.paper_safe_margin + self.offset[0] + i * (self.card_bbox_size + self.cut_lines_thickness)
            x_bottom = x_top + self.card_bbox_size
            yield x_top, 0, x_top, self.paper_size[1]
            yield x_bottom, 0, x_bottom, self.paper_size[1]


class FPDF2CardAssembler(CardAssembler):
    """Class for assembling cards into sheets using FPDF."""

    def assemble(self, card_image_filepaths: list[str | Path], output_filepath: str | Path):
        total_cards = len(card_image_filepaths)
        pages = np.ceil(total_cards / (self.rows * self.cols)).astype(int)
        tqdm_ = tqdm(total=total_cards, desc="Plotting cards")

        logger.info(f"Will print {total_cards} cards in {pages} pages in a {self.rows}x{self.cols} grid.")

        # Ensure directory exists
        output_filepath = Path(output_filepath)
        output_filepath.parent.mkdir(parents=True, exist_ok=True)

        # Initialize PDF
        pdf = FPDF(orientation="P", unit=self.units, format=self.paper_size)  # type: ignore

        for page_idx, bbox_gen in enumerate(self.get_page_generators(card_image_filepaths)):
            tqdm_.set_description(f"Plotting cards (page {page_idx + 1}/{pages})")
            pdf.add_page()
            if self.background_color is not None:
                pdf.set_fill_color(*self.background_color)
                pdf.rect(0, 0, float(self.paper_size[0]), float(self.paper_size[1]), "F")

            if self.crop_marks_thickness > 0.0:
                tqdm_.set_description(f"Plotting crop marks (page {page_idx + 1}/{pages})")
                pdf.set_line_width(self.crop_marks_thickness)
                for line_coordinates in self.get_line_generator():
                    pdf.line(*line_coordinates)

            tqdm_.set_description(f"Plotting cards (page {page_idx + 1}/{pages})")
            for bbox, image in bbox_gen:
                pdf.image(name=PIL.Image.fromarray(image), x=bbox[0], y=bbox[1], w=bbox[2], h=bbox[3])
                tqdm_.update(1)

        tqdm.write(f"Writing to {output_filepath}")
        pdf.output(str(output_filepath))






