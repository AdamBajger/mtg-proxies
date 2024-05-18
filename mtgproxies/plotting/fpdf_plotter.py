# standard imports
from __future__ import annotations

from typing import TYPE_CHECKING

# third-party imports
from fpdf import FPDF
from tqdm import tqdm

# local imports
from mtgproxies.plotting import CardAssembler


if TYPE_CHECKING:
    from pathlib import Path


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
