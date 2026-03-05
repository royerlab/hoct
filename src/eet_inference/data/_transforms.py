# Heavily influenced by Trackastra's WRFeatures:
# https://github.com/weigertlab/trackastra/blob/main/trackastra/data/wrfeat.py

import abc
import math
from collections.abc import Sequence

import numpy as np
import polars as pl
import torch

from eet_inference.data._batching import DataKeys


def _uniform(size: int, support: tuple[float, float]) -> torch.Tensor:
    return torch.rand(size) * (support[1] - support[0]) + support[0]


class BaseTransform(abc.ABC):
    @abc.abstractmethod
    def __call__(self, df: pl.DataFrame) -> pl.DataFrame:
        pass

    @abc.abstractmethod
    def __repr__(self) -> str:
        pass

    def __hash__(self) -> int:
        return hash(str(self))


class Flip(BaseTransform):
    def __init__(self, p: float = 0.5, columns: Sequence[str] = ("z", "y", "x")):
        self._p = p
        self._columns = columns

    def __call__(self, df: pl.DataFrame) -> pl.DataFrame:
        return df.with_columns(
            *[
                (2 * pl.col(c).mean() - pl.col(c)).alias(c)
                for c in self._columns
                if c in df.columns and torch.rand(1) < self._p
            ],
        )

    def __repr__(self) -> str:
        return f"Flip(p={self._p}, columns={self._columns})"


class Translate(BaseTransform):
    def __init__(self, values: list[int | float], columns: Sequence[str]):
        self._values = values
        self._columns = columns

    def __call__(self, df: pl.DataFrame) -> pl.DataFrame:
        return df.with_columns(*[(pl.col(c) + v).alias(c) for v, c in zip(self._values, self._columns, strict=True)])

    def __repr__(self) -> str:
        return f"Translate(values={self._values}, columns={self._columns})"


class Affine(BaseTransform):
    # Reference: https://github.com/weigertlab/trackastra/blob/00c419cf031f266b2d501e656b607416a8acfa46/trackastra/data/wrfeat.py#L388
    _ignored = (
        "t",
        "node_id",
        "cc",
        "track_id",
        "intensity_mean",
        "intensity_std",
        "intensity_min",
        "intensity_max",
        "border_dist",
    )

    def __init__(
        self,
        degree_range: tuple[float, float] | None,
        scale_range: tuple[float, float] | None,
        shear_range: tuple[tuple[float, float], tuple[float, float]] | None,
    ):
        self._degree_range = degree_range or (0, 0)
        self._scale_range = scale_range or (1, 1)
        self._shear_range = shear_range or ((0, 0), (0, 0))

    @staticmethod
    def _rotation_matrix(rad: float) -> np.ndarray:
        return np.asarray(
            [
                [1, 0, 0],
                [0, math.cos(rad), -math.sin(rad)],
                [0, math.sin(rad), math.cos(rad)],
            ]
        )

    @staticmethod
    def _shear_matrix(shear_y: float, shear_x: float) -> np.ndarray:
        return np.asarray(
            [
                [1, 0, 0],
                [0, 1 + shear_y * shear_x, shear_y],
                [0, shear_x, 1],
            ]
        )

    @staticmethod
    def _apply_affine_per_column(df: pl.DataFrame, affine: torch.Tensor, column: str | tuple[str, ...]) -> pl.DataFrame:
        if isinstance(column, tuple):
            data = df.select(column).to_numpy() @ affine
            return df.with_columns(
                *[pl.Series(name=c, values=data[:, i]) for i, c in enumerate(column)],
            )

        expr = pl.col(column)

        match column:
            case value if value in Affine._ignored:
                return df
            case "area":
                expr = np.linalg.det(affine) * pl.col("area")
            case "equivalent_diameter_area":
                expr = np.linalg.det(affine) ** (1 / len(affine)) * pl.col("equivalent_diameter_area")
            case "inertia_tensor":
                # out = M @ v @ Mt per row
                values = df[column].to_numpy()
                # v @ A'
                values = np.einsum("ijk, mk -> ijm", values, affine)
                # A @ v
                values = np.einsum("ij, kjm -> kim", affine, values)
                expr = pl.Series(name=column, values=values)
            case _:
                pass

        df = df.with_columns(expr.alias(column))

        return df

    def __call__(self, df: pl.DataFrame) -> pl.DataFrame:
        degrees = _uniform(1, self._degree_range).item()
        rad = np.deg2rad(degrees)
        scales = _uniform(3, self._scale_range)
        shear_y = _uniform(1, self._shear_range[0]).item()
        shear_x = _uniform(1, self._shear_range[1]).item()

        if "z" in df.columns:
            sp_columns = ["z", "y", "x"]
            ndim = 3
        else:
            sp_columns = ["y", "x"]
            ndim = 2

        affine = self._rotation_matrix(rad) @ np.diag(scales) @ self._shear_matrix(shear_y, shear_x)
        # original affine is 3D
        affine = affine[-ndim:, -ndim:]

        coords = df.select(sp_columns).to_numpy()

        prev_min = coords.min(axis=0, keepdims=True)
        mean = coords.mean(axis=0, keepdims=True)
        # centering before affine
        coords = coords - mean
        coords = coords @ affine + mean
        coords = coords - coords.min(axis=0, keepdims=True) + prev_min

        df = df.with_columns(
            *[pl.Series(name=c, values=coords[:, i]) for i, c in enumerate(sp_columns)],
        )

        for column in df.columns:
            df = self._apply_affine_per_column(df, affine, column)

        # NOTE: worst performance
        # spatial_columns = ("z", "y", "x") if "z" in df.columns else ("y", "x")
        # df = self._apply_affine_per_column(df, affine, spatial_columns)

        return df

    def __repr__(self) -> str:
        return (
            f"Affine(\n\tdegree_range={self._degree_range},\n\t"
            f"scale_range={self._scale_range},\n\tshear_range={self._shear_range}\n)"
        )


class Power(BaseTransform):
    def __init__(
        self,
        columns: Sequence[str],
        power_range: tuple[float, float],
    ):
        self._columns = columns
        self._power_range = power_range

    def __call__(self, df: pl.DataFrame) -> pl.DataFrame:
        exponent = _uniform(1, self._power_range).item()
        return df.with_columns(
            *[(pl.col(c).pow(exponent)).alias(c) for c in self._columns if c in df.columns],
        )

    def __repr__(self) -> str:
        return f"Power(columns={self._columns}, power_range={self._power_range})"


class Standardize:
    def __init__(self, mean: torch.Tensor, std: torch.Tensor):
        self._mean = torch.tensor(mean)
        self._std = torch.tensor(std).clamp(min=1e-7)

    def __call__(self, dict_data: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        dict_data[DataKeys.NODE_FEATS] = (dict_data[DataKeys.NODE_FEATS] - self._mean) / self._std
        return dict_data

    def __repr__(self) -> str:
        return f"StandardizeInput(mean={self._mean}, std={self._std})"
