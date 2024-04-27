import argparse
from pathlib import Path
from typing import IO, TextIO, Literal, Any

import click
import numpy as np
from nptyping import NDArray, Shape, Float32
from webcolors import name_to_rgb, IntegerRGB

from mtgproxies import fetch_scans_scryfall, print_cards_fpdf, print_cards_matplotlib
from mtgproxies.cli import parse_decklist_spec
from mtgproxies.decklists import Decklist
from mtgproxies.dimensions import (
    PAPER_SIZE,
    MTG_CARD_MM, Units, UNITS_TO_MM
)
from mtgproxies.print_cards import FPDF2CardAssembler


def papersize(string: str) -> np.ndarray:
    spec = string.lower()
    if spec == "a4":
        return np.array([21, 29.7]) / 2.54
    if "x" in spec:
        split = spec.split("x")
        return np.array([float(split[0]), float(split[1])])
    raise argparse.ArgumentTypeError()


def papersize_click_option(
        ctx: click.Context,
        param: click.Parameter,
        value: str
) -> str | NDArray[Shape["2"], Float32]:
    spec = value.upper()
    if spec in PAPER_SIZE:
        return spec
    elif "x" in spec:
        split = spec.split("x")
        if len(split) == 2:
            return np.asarray([float(split[0]), float(split[1])] , dtype=float)
        else:
            raise click.BadParameter(
                "Paper size must be in the format WIDTHxHEIGHT", param=param.name, ctx=ctx)

    else:
        raise click.BadParameter(
            f"Paper size not supported: {spec}. Try one of {PAPER_SIZE.keys()}"
            f"or define the dimensions in a WIDTHxHEIGHT format", param=param.name, ctx=ctx)


@click.group(name="print")
@click.pass_context
def ur_mom(ctx):
    ctx.ensure_object(dict)


def common_cli_arguments(func):
    func = click.argument(
        "output_file",
        type=click.Path(path_type=Path, exists=False, writable=True),
        required=True
    )(func)
    func = click.argument("deck_list", type=str, nargs=-1)(func)
    func = click.option(
        "--crop-mark-thickness", "-cm",
        type=float, default=0.0,
        help="Thickness of crop marks in the specified units. Use 0 to disable crop marks.",
    )(func)
    func = click.option(
        "--cut-lines-thickness", "-cl",
        type=float, default=0.0,
        help="Thickness of cut lines in the specified units. Use 0 to disable cut lines."
    )(func)
    func = click.option(
        "--crop-border", "-cb",
        type=float, default=0.0,
        help="How much to crop the borders of the cards in the specified units."
    )(func)
    func = click.option(
        "--background-color", "-bg",
        type=name_to_rgb, default=None,
        help="Background color of the cards, either by name or by hex code."
    )(func)
    func = click.option(
        "--paper-size", "-ps",
        type=str, default="a4", callback=papersize_click_option,
        help="Paper size keyword (A0 - A10) or dimensions in the format WIDTHxHEIGHT."
    )(func)
    func = click.option(
        "--page-safe-margin", "-m",
        type=float, default=0.0,
        help="Margin around the area where no cards will be printed. Useful for printers that can't print to the edge."
    )(func)
    func = click.option(
        "--faces", "-f",
        type=click.Choice(["all", "front", "back"]), default="all",
        help="Which faces to print."
    )(func)
    func = click.option(
        "--units", "-u",
        type=click.Choice(Units.__args__), default="mm",
        help="Units of the specified dimensions. Default is mm."
    )(func)
    func = click.option(
        "--fill-corners", "-fc",
        is_flag=True,
        help="Fill the corners of the cards with the colors of the closest pixels."
    )(func)
    func = click.option(
        "--cache-dir", "-cd",
        type=click.Path(path_type=Path, file_okay=False, dir_okay=True, writable=True),
        default=Path.cwd() / ".cache" / "mtgproxies",
        help="Directory to store cached card images."
    )(func)
    return func


@ur_mom.command(name="pdf")
@common_cli_arguments
@click.pass_context
def my_mom(
        ctx: click.Context,
        deck_list: list[str],
        output_file: Path,
        faces: Literal["all", "front", "back"],
        crop_mark_thickness: float,
        cut_lines_thickness: float,
        crop_border: float,
        background_color: IntegerRGB,
        paper_size: str | NDArray[Shape["2"], Float32],
        units: Units,
        fill_corners: bool,
        page_safe_margin: float,
        cache_dir: Path
):
    """This command generates a PDF document at OUTPUT_FILE with the cards from the files in DECK_LIST

    DECK_LIST is a list of files containing filepaths to decklist files in text/arena format
    or entries in a manastack:{manastack_id} or archidekt:{archidekt_id} format.

    OUTPUT_FILE is the path to the output PDF file.

    """
    parsed_deck_list = Decklist()
    for deck in deck_list:
        parsed_deck_list.extend(parse_decklist_spec(deck))

    # Fetch scans
    images = fetch_scans_scryfall(decklist=parsed_deck_list, faces=faces)

    # resolve paper size
    if isinstance(paper_size, str):
        if units in PAPER_SIZE[paper_size]:
            resolved_paper_size = PAPER_SIZE[paper_size][units]
        else:
            resolved_paper_size = PAPER_SIZE[paper_size]["mm"] / UNITS_TO_MM[units]
    else:
        resolved_paper_size = paper_size
    # Plot cards
    printer = FPDF2CardAssembler(
        units=units,
        paper_size=resolved_paper_size,
        card_size=MTG_CARD_MM,
        crop_marks_thickness=crop_mark_thickness,
        cut_lines_thickness=cut_lines_thickness,
        border_crop=crop_border,
        background_color=background_color,
        fill_corners=fill_corners,
        page_safe_margin=page_safe_margin
    )

    printer.assemble(card_image_filepaths=images, output_filepath=output_file)


if __name__ == "__main__":
    ur_mom(obj={})


