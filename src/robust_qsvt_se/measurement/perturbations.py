from __future__ import annotations

from typing import Any

import numpy as np

from robust_qsvt_se.measurement.linear_system import WeightedSystem

VALID_BAD_DATA_TARGETS = {"random", "weak_area"}


def add_gaussian_noise(
    system: WeightedSystem,
    *,
    noise_std: float,
    rng: np.random.Generator,
) -> WeightedSystem:
    if noise_std < 0.0:
        raise ValueError("noise_std must be nonnegative")
    if noise_std == 0.0:
        return system

    noise = rng.normal(loc=0.0, scale=noise_std, size=system.r_tilde.shape)
    metadata = {
        **system.metadata,
        "noise_std": float(noise_std),
        "noise_l2_norm": float(np.linalg.norm(noise)),
    }
    return WeightedSystem(
        H_tilde=system.H_tilde,
        r_tilde=system.r_tilde + noise,
        x_true=system.x_true,
        metadata=metadata,
    )


def remove_random_rows(
    system: WeightedSystem,
    *,
    missing_ratio: float,
    rng: np.random.Generator,
) -> WeightedSystem:
    if not 0.0 <= missing_ratio < 1.0:
        raise ValueError("missing_ratio must satisfy 0 <= missing_ratio < 1")
    n_drop = round(system.n_measurements * missing_ratio)
    if n_drop == 0:
        return system
    if system.n_measurements - n_drop < system.n_states:
        raise ValueError("missing_ratio leaves fewer rows than states")

    dropped_rows = np.sort(rng.choice(system.n_measurements, size=n_drop, replace=False))
    keep_mask = np.ones(system.n_measurements, dtype=bool)
    keep_mask[dropped_rows] = False
    metadata = _filter_row_metadata(system.metadata, keep_mask, system.n_measurements)
    metadata.update(
        {
            "missing_ratio": float(missing_ratio),
            "dropped_rows": dropped_rows.astype(int).tolist(),
        }
    )
    return WeightedSystem(
        H_tilde=system.H_tilde[keep_mask, :],
        r_tilde=system.r_tilde[keep_mask],
        x_true=system.x_true,
        metadata=metadata,
    )


def add_bad_data_outliers(
    system: WeightedSystem,
    *,
    bad_data_config: dict[str, Any] | None,
    rng: np.random.Generator,
) -> WeightedSystem:
    settings = _bad_data_settings(bad_data_config)
    rows, signs, outliers = _bad_data_outliers(
        n_rows=system.n_measurements,
        measurement_stds=None,
        metadata=system.metadata,
        settings=settings,
        rng=rng,
    )
    metadata = _bad_data_metadata(
        system.metadata,
        settings=settings,
        rows=rows,
        signs=signs,
        outliers=outliers,
    )
    if rows.size == 0:
        return WeightedSystem(
            H_tilde=system.H_tilde,
            r_tilde=system.r_tilde,
            x_true=system.x_true,
            metadata=metadata,
        )

    r_tilde = system.r_tilde.copy()
    r_tilde[rows] += outliers
    return WeightedSystem(
        H_tilde=system.H_tilde,
        r_tilde=r_tilde,
        x_true=system.x_true,
        metadata=metadata,
    )


def add_bad_data_to_measurements(
    z: np.ndarray,
    *,
    measurement_stds: np.ndarray,
    metadata: dict[str, Any],
    bad_data_config: dict[str, Any] | None,
    rng: np.random.Generator,
) -> tuple[np.ndarray, dict[str, Any]]:
    values = np.asarray(z, dtype=np.float64)
    stds = np.asarray(measurement_stds, dtype=np.float64)
    if values.shape != stds.shape:
        raise ValueError("measurement_stds must match z shape")
    settings = _bad_data_settings(bad_data_config)
    rows, signs, outliers = _bad_data_outliers(
        n_rows=values.size,
        measurement_stds=stds,
        metadata=metadata,
        settings=settings,
        rng=rng,
    )
    updated = values.copy()
    if rows.size:
        updated[rows] += outliers
    return (
        updated,
        _bad_data_metadata(
            metadata,
            settings=settings,
            rows=rows,
            signs=signs,
            outliers=outliers,
        ),
    )


def _filter_row_metadata(
    metadata: dict[str, Any],
    keep_mask: np.ndarray,
    n_rows: int,
) -> dict[str, Any]:
    filtered = dict(metadata)
    keep_indices = np.flatnonzero(keep_mask)
    for key in ("measurement_types", "measurement_labels", "measurement_stds", "measurement_buses"):
        value = filtered.get(key)
        if isinstance(value, list) and len(value) == n_rows:
            filtered[key] = [value[index] for index in keep_indices]
    return filtered


def _bad_data_settings(config: dict[str, Any] | None) -> dict[str, Any]:
    config = config or {}
    settings = {
        "enabled": bool(config.get("enabled", False)),
        "ratio": float(config.get("ratio", 0.0)),
        "magnitude": float(config.get("magnitude", 10.0)),
        "target": str(config.get("target", "random")),
    }
    if not 0.0 <= settings["ratio"] < 1.0:
        raise ValueError("bad_data.ratio must satisfy 0 <= ratio < 1")
    if settings["magnitude"] <= 0.0:
        raise ValueError("bad_data.magnitude must be positive")
    if settings["target"] not in VALID_BAD_DATA_TARGETS:
        raise ValueError(f"bad_data.target must be one of {sorted(VALID_BAD_DATA_TARGETS)}")
    return settings


def _bad_data_outliers(
    *,
    n_rows: int,
    measurement_stds: np.ndarray | None,
    metadata: dict[str, Any],
    settings: dict[str, Any],
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not settings["enabled"] or settings["ratio"] == 0.0:
        empty_int = np.array([], dtype=int)
        return empty_int, empty_int, np.array([], dtype=np.float64)

    eligible_rows = _eligible_bad_data_rows(
        n_rows=n_rows,
        target=str(settings["target"]),
        measurement_buses=metadata.get("measurement_buses"),
        weak_area_buses=metadata.get("weak_area_buses"),
    )
    n_bad = round(len(eligible_rows) * float(settings["ratio"]))
    if n_bad == 0:
        empty_int = np.array([], dtype=int)
        return empty_int, empty_int, np.array([], dtype=np.float64)

    rows = np.sort(rng.choice(eligible_rows, size=n_bad, replace=False)).astype(int)
    signs = rng.choice(np.array([-1, 1], dtype=int), size=n_bad, replace=True)
    outliers = signs.astype(np.float64) * float(settings["magnitude"])
    if measurement_stds is not None:
        outliers = outliers * measurement_stds[rows]
    return rows, signs, outliers


def _eligible_bad_data_rows(
    *,
    n_rows: int,
    target: str,
    measurement_buses: Any,
    weak_area_buses: Any,
) -> np.ndarray:
    if target == "random":
        return np.arange(n_rows, dtype=int)

    if target != "weak_area":
        raise ValueError(f"bad_data.target must be one of {sorted(VALID_BAD_DATA_TARGETS)}")
    weak_area = {int(bus) for bus in weak_area_buses or ()}
    if not weak_area:
        raise ValueError("bad_data.target='weak_area' requires system.measurement.weak_area_buses")
    if not isinstance(measurement_buses, list) or len(measurement_buses) != n_rows:
        raise ValueError("bad_data.target='weak_area' requires per-row measurement_buses metadata")

    eligible = [
        index
        for index, buses in enumerate(measurement_buses)
        if weak_area.intersection(int(bus) for bus in buses)
    ]
    if not eligible:
        raise ValueError("bad_data.target='weak_area' found no eligible measurement rows")
    return np.array(eligible, dtype=int)


def _bad_data_metadata(
    metadata: dict[str, Any],
    *,
    settings: dict[str, Any],
    rows: np.ndarray,
    signs: np.ndarray,
    outliers: np.ndarray,
) -> dict[str, Any]:
    return {
        **metadata,
        "bad_data_enabled": bool(settings["enabled"]),
        "bad_data_ratio": float(settings["ratio"]),
        "bad_data_magnitude": float(settings["magnitude"]),
        "bad_data_target": str(settings["target"]),
        "bad_data_count": int(rows.size),
        "bad_data_rows": rows.astype(int).tolist(),
        "bad_data_signs": signs.astype(int).tolist(),
        "bad_data_l2_norm": float(np.linalg.norm(outliers)),
    }
