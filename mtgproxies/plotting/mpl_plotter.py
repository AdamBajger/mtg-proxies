# standard imports
from __future__ import annotations

# standard imports
import math
from typing import TYPE_CHECKING

# third-party imports
import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm

# local imports
from mtgproxies.dimensions import UNITS_TO_IN
from mtgproxies.plotting import CardAssembler


if TYPE_CHECKING:
    from pathlib import Path

    from mtgproxies.scans import CardSlot


class MatplotlibCardAssembler(CardAssembler):
    """Class for assembling cards into sheets using Matplotlib."""

    def __init__(self, dpi: int, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dpi = dpi

        self.paper_size = self.paper_size * UNITS_TO_IN[self.units]
        self.card_size = self.card_size * UNITS_TO_IN[self.units]
        self.card_bbox_size = self.card_bbox_size * UNITS_TO_IN[self.units]
        self.border_crop = self.border_crop * UNITS_TO_IN[self.units]
        self.crop_marks_thickness = self.crop_marks_thickness * UNITS_TO_IN[self.units]
        self.cut_spacing_thickness = self.cut_spacing_thickness * UNITS_TO_IN[self.units]
        self.paper_safe_margin = self.paper_safe_margin * UNITS_TO_IN[self.units]
        self.offset = self.offset * UNITS_TO_IN[self.units]
        self.safe_printable_area = self.safe_printable_area * UNITS_TO_IN[self.units]
        self.grid_bbox_size = self.grid_bbox_size * UNITS_TO_IN[self.units]

    def assemble(self, card_image_filepaths: list[Path], output_filepath: Path):
        pages, tqdm_ = self.prepare_routine(card_image_filepaths, output_filepath)
        digits = int(np.ceil(math.log10(pages)))

        with tqdm(total=len(card_image_filepaths), desc="Plotting cards") as pbar:
            for page_idx, bbox_gen in enumerate(self.get_page_generators(card_image_filepaths)):
                tqdm_.set_description(f"Plotting cards (page {page_idx + 1}/{pages})")
                fig = plt.figure(figsize=self.paper_size)
                ax = fig.add_axes(rect=(0, 0, 1, 1))
                ax.set_xlim(0, 1)
                ax.set_ylim(0, 1)
                ax.invert_yaxis()

                if self.crop_marks_thickness > 0.0:
                    pbar.set_description("plotting crop marks...")
                    crop_marks_thickness_in_pt = self.crop_marks_thickness * 72
                    for line_coordinates in self.get_line_generator():
                        x0, y0, x1, y1 = line_coordinates
                        x_rel = np.asarray([x0, x1]) / self.paper_size[0]
                        y_rel = np.asarray([y0, y1]) / self.paper_size[1]

                        # convert cropmarks thickness to point units
                        ax.plot(x_rel, y_rel, color="black", linewidth=crop_marks_thickness_in_pt)

                for bbox, image in bbox_gen:
                    left, top, width, height = bbox

                    x0 = left / self.paper_size[0]
                    y0 = top / self.paper_size[1]

                    width_scaled = width / self.paper_size[0]
                    height_scaled = height / self.paper_size[1]

                    x1 = x0 + width_scaled
                    y1 = y0 + height_scaled

                    # extent = (left, right, bottom, top)
                    extent = (x0, x1, y0, y1)

                    _ = ax.imshow(image, extent=extent, interpolation="lanczos", aspect="auto", origin="lower")
                    pbar.update(1)

                # save the page and skip the rest
                out_file_name = (
                    output_filepath.parent / f"{output_filepath.stem}_{page_idx:0{digits}d}{output_filepath.suffix}"
                )
                fig.savefig(fname=out_file_name, dpi=self.dpi)
                plt.close()

    def assemble_duplex(
        self,
        card_slots: list[CardSlot],
        output_filepath: Path,
        flip_mode: str = CardAssembler.FLIP_HORIZONTAL,
    ):
        """Assemble cards for double-sided (duplex) printing as image files.

        Each page pair is saved as two image files: ``stem_f0.ext`` (front) and
        ``stem_b0.ext`` (back), etc.

        Args:
            card_slots: One :class:`~mtgproxies.scans.CardSlot` per card copy.
            output_filepath: Base path for output images.
            flip_mode: ``FLIP_HORIZONTAL`` (default) or ``FLIP_VERTICAL``.
        """
        total_slots = len(card_slots)
        page_pairs = int(np.ceil(total_slots / (self.rows * self.cols)))
        digits = max(2, int(np.ceil(math.log10(page_pairs))))
        tqdm_ = tqdm(total=total_slots * 2, desc="Plotting cards")
        output_filepath.parent.mkdir(parents=True, exist_ok=True)

        pair_idx = 0
        for front_layout, back_gen in self.get_duplex_page_pairs(card_slots, flip_mode):

            # --- front page ---
            tqdm_.set_description(f"Plotting cards (front page {pair_idx}/{page_pairs})")
            fig = plt.figure(figsize=self.paper_size)
            ax = fig.add_axes(rect=(0, 0, 1, 1))
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.invert_yaxis()
            if self.crop_marks_thickness > 0.0:
                crop_marks_thickness_in_pt = self.crop_marks_thickness * 72
                for line_coordinates in self.get_line_generator():
                    x0, y0, x1, y1 = line_coordinates
                    ax.plot(np.asarray([x0, x1]) / self.paper_size[0],
                            np.asarray([y0, y1]) / self.paper_size[1],
                            color="black", linewidth=crop_marks_thickness_in_pt)
            for bbox, image, _is_double in front_layout:
                left, top, width, height = bbox
                extent = (left / self.paper_size[0], (left + width) / self.paper_size[0],
                          top / self.paper_size[1], (top + height) / self.paper_size[1])
                ax.imshow(image, extent=extent, interpolation="lanczos", aspect="auto", origin="lower")
                tqdm_.update(1)
            front_name = output_filepath.parent / f"{output_filepath.stem}_f{pair_idx:0{digits}d}{output_filepath.suffix}"
            fig.savefig(fname=front_name, dpi=self.dpi)
            plt.close()

            # --- back page ---
            tqdm_.set_description(f"Plotting cards (back page {pair_idx}/{page_pairs})")
            fig = plt.figure(figsize=self.paper_size)
            ax = fig.add_axes(rect=(0, 0, 1, 1))
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.invert_yaxis()
            if self.crop_marks_thickness > 0.0:
                crop_marks_thickness_in_pt = self.crop_marks_thickness * 72
                for line_coordinates in self.get_line_generator():
                    x0, y0, x1, y1 = line_coordinates
                    ax.plot(np.asarray([x0, x1]) / self.paper_size[0],
                            np.asarray([y0, y1]) / self.paper_size[1],
                            color="black", linewidth=crop_marks_thickness_in_pt)
            for bbox, image in back_gen:
                left, top, width, height = bbox
                extent = (left / self.paper_size[0], (left + width) / self.paper_size[0],
                          top / self.paper_size[1], (top + height) / self.paper_size[1])
                ax.imshow(image, extent=extent, interpolation="lanczos", aspect="auto", origin="lower")
                tqdm_.update(1)
            back_name = output_filepath.parent / f"{output_filepath.stem}_b{pair_idx:0{digits}d}{output_filepath.suffix}"
            fig.savefig(fname=back_name, dpi=self.dpi)
            plt.close()

            pair_idx += 1
