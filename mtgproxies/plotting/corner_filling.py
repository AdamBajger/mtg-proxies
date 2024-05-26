from __future__ import annotations

from typing import TYPE_CHECKING

import PIL
from PIL import ImageChops, ImageFilter
from PIL.Image import Image, Transpose


if TYPE_CHECKING:
    from typing import Literal


def blend_patch_into_image(bbox: tuple[int, int, int, int], image: PIL.Image, patch: PIL.Image) -> PIL.Image:
    """Blends an image patch into another image at the defined bounding box using an alpha mask.

    The patch is inserted over the image based on the values in the alpha channel of the image.
    The pixels are pasted only to fill in the transparent regions in the image (as described by the alpha channel)

    The edges of the original card are eroded away slightly to remove any black lining that would disturb the blend.
    Then, the edge is blurred a little to improve the blending of the pixels that are filled in.

    Args:
        bbox (tuple[int, int, int, int]): The bounding box that further limits the area where pixels are being written
        image (PIL.Image): The image to be written over.
        patch (PIL.Image): The image patch to be pasted over the base image.

    Returns:
        A PIL image with the result
    """
    size = min(image.size)
    alpha = image.split()[3]
    alpha = ImageChops.invert(alpha)
    dill_size = size // 10 + 1 - (size // 10 % 2)  # values identified manually
    blur_size = size // 30  # values identified manually
    alpha = alpha.filter(ImageFilter.MaxFilter(dill_size))
    alpha = alpha.filter(ImageFilter.GaussianBlur(blur_size))
    # paste the blurred image where the alpha is 255
    image.paste(patch, mask=alpha.crop(bbox), box=bbox)
    return image


def blend_flipped_stripe(
    square_image: Image,
    stripe_width_fraction: float,
    flip_how: Literal["horizontal", "vertical"],
    stripe_location: Literal["top", "bottom", "left", "right"],
) -> Image:
    """Takes a leftmost stripe of a square image, flips it, and blends it back into the image.

    The stripe is blended using the alpha channel of the image to fill in the transparent regions
    with the non-transparent regions of the flipped stripe.
    Basically the stripe is flipped and then pasted back into the image only filling in the transparent regions.

    Args:
        square_image: Image to process.
        stripe_width_fraction: Fraction of the width/height of the square image to use as the stripe.
        flip_how: How to flip the stripe. Either "horizontal" or "vertical".
        stripe_location: Where to take the stripe from. Either "top", "bottom", "left", or "right" of the square
                         image. The stripe is adjacent to the edge.

    Returns:
        The image with the flipped stripe blended in.
    """
    corner_copy = square_image.copy()
    width, height = corner_copy.size
    if stripe_location in ["top", "bottom"]:
        transpose_method = Transpose.FLIP_LEFT_RIGHT
    elif stripe_location in ["left", "right"]:
        transpose_method = Transpose.FLIP_TOP_BOTTOM
    else:
        raise ValueError(f"Invalid stripe_location: {stripe_location}")

    if flip_how == "horizontal":
        if stripe_location == "top":
            stripe_width = int(height * stripe_width_fraction)
            bbox = (0, 0, width, stripe_width)
        elif stripe_location == "bottom":
            stripe_width = int(height * (1 - stripe_width_fraction))
            bbox = (0, stripe_width, width, height)
        else:
            raise ValueError(f"Invalid stripe_location: {stripe_location} for flip_how: {flip_how}")
    elif flip_how == "vertical":
        if stripe_location == "left":
            stripe_width = int(width * stripe_width_fraction)
            bbox = (0, 0, stripe_width, height)
        elif stripe_location == "right":
            stripe_width = int(width * (1 - stripe_width_fraction))
            bbox = (stripe_width, 0, width, height)
        else:
            raise ValueError(f"Invalid stripe_location: {stripe_location} for flip_how: {flip_how}")
    else:
        raise ValueError(f"Invalid flip_how: {flip_how}")

    patch_inverted = corner_copy.crop(bbox).transpose(method=transpose_method)
    return blend_patch_into_image(bbox, corner_copy, patch_inverted)


def fill_corners(card_image: Image) -> Image:
    """Fill the corners of the card with the closest pixels around the corners to match the border color."""
    corner_size = card_image.width // 10

    # top corners, vertical stripes
    box_left = (0, 0, corner_size, corner_size)
    card_image.paste(
        blend_flipped_stripe(
            square_image=card_image.crop(box=box_left),
            stripe_width_fraction=1 / 6,
            flip_how="vertical",
            stripe_location="left",
        ),
        box=box_left,
    )
    box_right = (card_image.width - corner_size, 0, card_image.width, corner_size)
    card_image.paste(
        blend_flipped_stripe(
            square_image=card_image.crop(box=box_right),
            stripe_width_fraction=1 / 6,
            flip_how="vertical",
            stripe_location="right",
        ),
        box=box_right,
    )

    # bottom corners, vertical stripes
    box_left = (0, card_image.height - corner_size, corner_size, card_image.height)
    card_image.paste(
        blend_flipped_stripe(
            square_image=card_image.crop(box=box_left),
            stripe_width_fraction=1 / 6,
            flip_how="vertical",
            stripe_location="left",
        ),
        box=box_left,
    )
    box_right = (card_image.width - corner_size, card_image.height - corner_size, card_image.width, card_image.height)
    card_image.paste(
        blend_flipped_stripe(
            square_image=card_image.crop(box=box_right),
            stripe_width_fraction=1 / 6,
            flip_how="vertical",
            stripe_location="right",
        ),
        box=box_right,
    )

    # top corners, horizontal stripes
    box_top = (0, 0, corner_size, corner_size)
    card_image.paste(
        blend_flipped_stripe(
            square_image=card_image.crop(box=box_top),
            stripe_width_fraction=1 / 6,
            flip_how="horizontal",
            stripe_location="top",
        ),
        box=box_top,
    )
    box_bottom = (card_image.width - corner_size, 0, card_image.width, corner_size)
    card_image.paste(
        blend_flipped_stripe(
            square_image=card_image.crop(box=box_bottom),
            stripe_width_fraction=1 / 6,
            flip_how="horizontal",
            stripe_location="top",
        ),
        box=box_bottom,
    )

    # bottom corners, horizontal stripes
    box_top = (0, card_image.height - corner_size, corner_size, card_image.height)
    card_image.paste(
        blend_flipped_stripe(
            square_image=card_image.crop(box=box_top),
            stripe_width_fraction=1 / 6,
            flip_how="horizontal",
            stripe_location="bottom",
        ),
        box=box_top,
    )
    box_bottom = (card_image.width - corner_size, card_image.height - corner_size, card_image.width, card_image.height)
    card_image.paste(
        blend_flipped_stripe(
            square_image=card_image.crop(box=box_bottom),
            stripe_width_fraction=1 / 6,
            flip_how="horizontal",
            stripe_location="bottom",
        ),
        box=box_bottom,
    )

    return card_image
