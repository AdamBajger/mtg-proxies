import numpy as np
from nptyping import NDArray, Float, Int


MTG_CARD_INCHES: NDArray[2, Float] = np.asarray([2.48, 3.46], dtype=float)
MTG_CARD_MM: NDArray[2, Float] = np.asarray([62.992, 87.884], dtype=float)

# Paper sizes (sourced from the Adobe website)
PAPER_A0_MM: NDArray[2, Float] = np.asarray([841, 1189], dtype=float)
PAPER_A1_MM: NDArray[2, Float] = np.asarray([594, 841], dtype=float)
PAPER_A2_MM: NDArray[2, Float] = np.asarray([420, 594], dtype=float)
PAPER_A3_MM: NDArray[2, Float] = np.asarray([297, 420], dtype=float)
PAPER_A4_MM: NDArray[2, Float] = np.asarray([210, 297], dtype=float)
PAPER_A5_MM: NDArray[2, Float] = np.asarray([148, 210], dtype=float)
PAPER_A6_MM: NDArray[2, Float] = np.asarray([105, 148], dtype=float)
PAPER_A7_MM: NDArray[2, Float] = np.asarray([74, 105], dtype=float)
PAPER_A8_MM: NDArray[2, Float] = np.asarray([52, 74], dtype=float)
PAPER_A9_MM: NDArray[2, Float] = np.asarray([37, 52], dtype=float)
PAPER_A10_MM: NDArray[2, Float] = np.asarray([26, 37], dtype=float)

PAPER_A0_INCHES: NDArray[2, Float] = np.asarray([33.1, 46.8], dtype=float)
PAPER_A1_INCHES: NDArray[2, Float] = np.asarray([23.4, 33.1], dtype=float)
PAPER_A2_INCHES: NDArray[2, Float] = np.asarray([16.5, 23.4], dtype=float)
PAPER_A3_INCHES: NDArray[2, Float] = np.asarray([11.7, 16.5], dtype=float)
PAPER_A4_INCHES: NDArray[2, Float] = np.asarray([8.3, 11.7], dtype=float)
PAPER_A5_INCHES: NDArray[2, Float] = np.asarray([5.8, 8.3], dtype=float)
PAPER_A6_INCHES: NDArray[2, Float] = np.asarray([4.1, 5.8], dtype=float)
PAPER_A7_INCHES: NDArray[2, Float] = np.asarray([2.9, 4.1], dtype=float)
PAPER_A8_INCHES: NDArray[2, Float] = np.asarray([2.0, 2.9], dtype=float)
PAPER_A9_INCHES: NDArray[2, Float] = np.asarray([1.5, 2.0], dtype=float)
PAPER_A10_INCHES: NDArray[2, Float] = np.asarray([1.0, 1.5], dtype=float)


def get_pixels_for_dpi(dpi: int, area: NDArray[2, Int] = MTG_CARD_INCHES) -> NDArray[2, Int]:
    return (dpi * area).round(decimals=0).astype(int)


