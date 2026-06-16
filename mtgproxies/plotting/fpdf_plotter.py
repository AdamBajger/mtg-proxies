# standard imports
from __future__ import annotations

from typing import TYPE_CHECKING

# third-party imports
import numpy as np
from fpdf import FPDF
from tqdm import tqdm

# local imports
from mtgproxies.plotting import CardAssembler


if TYPE_CHECKING:
    from pathlib import Path

    from mtgproxies.scans import CardSlot


class FPDF2CardAssembler(CardAssembler):
    """Class for assembling cards into sheets using FPDF."""

    def assemble(self, card_image_filepaths: list[Path], output_filepath: Path):
        pages, tqdm_ = self.prepare_routine(card_image_filepaths, output_filepath)

        # Initialize PDF
        pdf = FPDF(orientation="P", unit=self.units, format=self.paper_size)

        for page_idx, bbox_gen in enumerate(self.get_page_generators(card_image_filepaths)):
            tqdm_.set_description(f"Plotting cards (page {page_idx + 1}/{pages})")
            pdf.add_page()
            if self.background_color is not None:
                pdf.set_fill_color(*self.background_color)
                pdf.rect(0, 0, float(self.paper_size[0]), float(self.paper_size[1]), "F")

            # print crop marks
            if self.crop_marks_thickness is not None and self.crop_marks_thickness > 0.0:
                tqdm_.set_description(f"Plotting crop marks (page {page_idx + 1}/{pages})")
                pdf.set_line_width(self.crop_marks_thickness)
                for line_coordinates in self.get_line_generator():
                    pdf.line(*line_coordinates)

            # print cards
            tqdm_.set_description(f"Plotting cards (page {page_idx + 1}/{pages})")
            for bbox, image in bbox_gen:
                pdf.image(name=image, x=bbox[0], y=bbox[1], w=bbox[2], h=bbox[3])
                tqdm_.update(1)

        tqdm.write(f"Writing to {output_filepath}")
        pdf.output(str(output_filepath))

    def assemble_duplex(
        self,
        card_slots: list[CardSlot],
        output_filepath: Path,
        flip_mode: str = CardAssembler.FLIP_HORIZONTAL,
    ):
        """Assemble cards for double-sided (duplex) printing.

        Generates paired front/back pages with double-faced card backs aligned
        to their corresponding fronts via long-edge (horizontal) flipping.

        Args:
            card_slots: One :class:`~mtgproxies.scans.CardSlot` per card copy.
            output_filepath: Path for the output PDF.
            flip_mode: ``FLIP_HORIZONTAL`` (default) or ``FLIP_VERTICAL``.
        """
        total_slots = len(card_slots)
        page_pairs = int(np.ceil(total_slots / (self.rows * self.cols)))
        tqdm_ = tqdm(total=total_slots * 2, desc="Plotting cards")
        output_filepath.parent.mkdir(parents=True, exist_ok=True)

        pdf = FPDF(orientation="P", unit=self.units, format=self.paper_size)
        pair_idx = 0
        for front_layout, back_gen in self.get_duplex_page_pairs(card_slots, flip_mode):
            pair_idx += 1
            has_crop_marks = self.crop_marks_thickness is not None and self.crop_marks_thickness > 0.0

            # --- front page ---
            tqdm_.set_description(f"Plotting cards (front page {pair_idx}/{page_pairs})")
            pdf.add_page()
            if self.background_color is not None:
                pdf.set_fill_color(*self.background_color)
                pdf.rect(0, 0, float(self.paper_size[0]), float(self.paper_size[1]), "F")
            if has_crop_marks:
                pdf.set_line_width(self.crop_marks_thickness)
                for line_coordinates in self.get_line_generator():
                    pdf.line(*line_coordinates)
            for bbox, image, _is_double in front_layout:
                pdf.image(name=image, x=bbox[0], y=bbox[1], w=bbox[2], h=bbox[3])
                tqdm_.update(1)

            # --- back page ---
            tqdm_.set_description(f"Plotting cards (back page {pair_idx}/{page_pairs})")
            pdf.add_page()
            if self.background_color is not None:
                pdf.set_fill_color(*self.background_color)
                pdf.rect(0, 0, float(self.paper_size[0]), float(self.paper_size[1]), "F")
            if has_crop_marks:
                pdf.set_line_width(self.crop_marks_thickness)
                for line_coordinates in self.get_line_generator():
                    pdf.line(*line_coordinates)
            for bbox, image in back_gen:
                pdf.image(name=image, x=bbox[0], y=bbox[1], w=bbox[2], h=bbox[3])
                tqdm_.update(1)

        tqdm.write(f"Writing to {output_filepath}")
        pdf.output(str(output_filepath))
