"""Tests for :class:`~extreme_weather.ingestion.netcdf.NetCDFSource`.

NetCDF is the project's native format, so this reader is the reference
implementation of the data-source interface. What is tested here is the set of
promises the rest of the ingestion layer relies on:

* a valid file opens, and opens **lazily** by default, because a real reanalysis
  file is tens of gigabytes;
* :meth:`open_context` always closes the file handle - on Windows an open handle
  blocks deletion, so a leaked handle is a real bug, not a style issue;
* every failure mode is a *specific* exception (missing file / wrong extension /
  unreadable bytes / path outside the base directory), never an empty dataset;
* ``fetch()`` returns the resolved absolute path, and the small helpers
  (``default_chunks``, ``file_timestamp``) behave as documented.

No network, no large binaries: every file is a 4x3x3 grid written by the tests
into ``tmp_path``.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import numpy as np
import pytest

from extreme_weather.ingestion import NETCDF_SUFFIXES, NetCDFSource, default_chunks
from extreme_weather.ingestion.exceptions import (
    FormatNotSupportedError,
    IngestionError,
    SourceNotFoundError,
)
from extreme_weather.ingestion.netcdf import file_timestamp
from tests.ingestion_fixtures import make_dataset, write_netcdf


@pytest.fixture()
def source() -> NetCDFSource:
    """An unconstrained reader, as the ingestion service builds it."""
    return NetCDFSource()


@pytest.fixture()
def good_file(tmp_path: Path) -> Path:
    """A small, valid, complete NetCDF file."""
    return write_netcdf(make_dataset(), tmp_path / "good.nc")


# --------------------------------------------------------------------------- #
# The source identity
# --------------------------------------------------------------------------- #
def test_netcdf_is_always_available_and_claims_no_credentials(source: NetCDFSource) -> None:
    """NetCDF is a core dependency, so it must never report itself unusable."""
    assert source.available() is True
    assert source.name() == "netcdf"
    assert source.default_origin() == "HISTORICAL"


def test_metadata_lists_every_supported_extension(source: NetCDFSource) -> None:
    info = source.metadata()
    assert info.formats == NETCDF_SUFFIXES
    assert ".nc" in info.formats and ".nc4" in info.formats
    assert info.requires_credentials is False


# --------------------------------------------------------------------------- #
# Loading a valid file
# --------------------------------------------------------------------------- #
def test_load_returns_the_variables_and_coordinates_of_the_file(
    source: NetCDFSource, good_file: Path
) -> None:
    with source.open_context(good_file) as dataset:
        assert set(dataset.data_vars) >= {"temperature", "precipitation", "u_wind", "v_wind"}
        assert dataset.sizes == {"time": 4, "latitude": 3, "longitude": 3}
        assert dataset["temperature"].attrs["units"] == "K"
        assert list(dataset.latitude.values) == [12.0, 12.5, 13.0]


def test_load_is_lazy_by_default_and_chunked_on_request(
    source: NetCDFSource, good_file: Path
) -> None:
    """A multi-gigabyte reanalysis file must not be read into memory on open.

    ``xr.open_dataset`` without ``chunks`` leaves the arrays backed by the open
    file; passing a chunk spec hands them to Dask instead. Both are asserted
    through the public result rather than through xarray internals.
    """
    lazy = source.load(good_file)
    try:
        # No Dask chunks: the values are read from the still-open file on access.
        assert lazy["temperature"].chunks is None
        assert "source" in lazy["temperature"].encoding
    finally:
        lazy.close()

    chunked = source.load(good_file, chunks=default_chunks(4))
    try:
        assert chunked["temperature"].chunks is not None
    finally:
        chunked.close()


def test_open_context_loads_the_values_and_releases_the_handle(
    source: NetCDFSource, tmp_path: Path
) -> None:
    """The handle must be closed, so the file can be deleted afterwards.

    On Windows an open NetCDF handle makes ``unlink`` fail with
    ``PermissionError``; on POSIX the file can be removed but the dataset's
    arrays would still be lazily bound. Deleting the file is therefore a direct,
    honest test of "the handle is gone".
    """
    path = write_netcdf(make_dataset(), tmp_path / "context.nc")
    with source.open_context(path) as dataset:
        assert dataset["temperature"].shape == (4, 3, 3)
        assert isinstance(dataset["temperature"].values, np.ndarray)

    path.unlink()
    assert not path.exists()


# --------------------------------------------------------------------------- #
# Failure modes - each one specific, never an empty dataset
# --------------------------------------------------------------------------- #
def test_a_corrupt_file_raises_ingestion_error(source: NetCDFSource, tmp_path: Path) -> None:
    """Bytes that are not a NetCDF file must fail loudly and name the file."""
    broken = tmp_path / "broken.nc"
    broken.write_text("this is definitely not a NetCDF file", encoding="utf-8")

    with pytest.raises(IngestionError) as info:
        source.load(broken)

    assert "broken.nc" in str(info.value)

    # A truncated real file is the more common real-world corruption.
    truncated = tmp_path / "truncated.nc"
    good = write_netcdf(make_dataset(), tmp_path / "src.nc")
    truncated.write_bytes(good.read_bytes()[:64])
    with pytest.raises(IngestionError):
        source.load(truncated)


def test_a_missing_file_raises_source_not_found(source: NetCDFSource, tmp_path: Path) -> None:
    with pytest.raises(SourceNotFoundError, match="not found"):
        source.load(tmp_path / "absent.nc")
    with pytest.raises(SourceNotFoundError, match="not found"):
        source.fetch(tmp_path / "absent.nc")


def test_a_wrong_extension_raises_format_not_supported(
    source: NetCDFSource, tmp_path: Path
) -> None:
    """An unsupported extension is a different problem from a missing file."""
    text = tmp_path / "notes.txt"
    text.write_text("hello", encoding="utf-8")

    with pytest.raises(FormatNotSupportedError) as info:
        source.fetch(text)

    assert ".nc" in str(info.value)


def test_a_directory_is_not_a_dataset(source: NetCDFSource, tmp_path: Path) -> None:
    folder = tmp_path / "looks_like.nc"
    folder.mkdir()
    with pytest.raises(SourceNotFoundError):
        source.fetch(folder)


@pytest.mark.parametrize("escape", ["../outside.nc", "../../outside.nc"])
def test_paths_outside_base_dir_are_rejected(tmp_path: Path, escape: str) -> None:
    """The base directory is a security boundary, not a convenience default.

    Without it, the ingest endpoint would read arbitrary files off the machine.
    """
    base = tmp_path / "raw"
    base.mkdir()
    write_netcdf(make_dataset(), tmp_path / "outside.nc")
    guarded = NetCDFSource(base_dir=base)

    with pytest.raises(SourceNotFoundError, match="outside the configured data directory"):
        guarded.fetch(escape)


def test_a_relative_path_inside_base_dir_resolves_against_it(tmp_path: Path) -> None:
    base = tmp_path / "raw"
    base.mkdir()
    write_netcdf(make_dataset(), base / "inside.nc")
    guarded = NetCDFSource(base_dir=base)

    resolved = guarded.fetch("inside.nc")

    assert resolved == (base / "inside.nc").resolve()
    assert resolved.is_absolute()


# --------------------------------------------------------------------------- #
# fetch() and the small helpers
# --------------------------------------------------------------------------- #
def test_fetch_returns_the_resolved_absolute_path(source: NetCDFSource, good_file: Path) -> None:
    """``fetch`` validates and returns; it never copies or rewrites the file."""
    resolved = source.fetch(good_file)

    assert resolved.is_absolute()
    assert resolved == good_file.resolve()
    # The bytes are untouched: fetch is a validation step, not a copy.
    assert resolved.read_bytes() == good_file.read_bytes()


def test_fetch_resolves_a_relative_path_against_base_dir_only(
    source: NetCDFSource, tmp_path: Path, good_file: Path
) -> None:
    """Relative paths are resolved against ``base_dir`` - or against the CWD.

    Without a ``base_dir`` there is nothing to resolve against, so a relative
    path is interpreted relative to the process working directory. That is why
    the ingestion service always builds its source with a ``base_dir``: it is
    what stops a request-supplied relative path from escaping the data folder.
    """
    guarded = NetCDFSource(base_dir=tmp_path)
    assert guarded.fetch(good_file.name) == good_file.resolve()

    with pytest.raises(SourceNotFoundError):
        source.fetch(good_file.name)


def test_default_chunks_keeps_a_day_of_hourly_data_in_one_chunk() -> None:
    """Short series are not over-chunked; long ones are capped at 24 steps."""
    assert default_chunks(4) == {"time": 4}
    assert default_chunks(24) == {"time": 24}
    assert default_chunks(72) == {"time": 24}
    # A zero/negative length must still yield a usable, non-zero chunk.
    assert default_chunks(0) == {"time": 1}
    assert default_chunks(-5) == {"time": 1}


def test_file_timestamp_is_an_iso_8601_utc_string(good_file: Path) -> None:
    stamp = file_timestamp(good_file)

    parsed = datetime.fromisoformat(stamp)
    assert parsed.tzinfo is not None
    assert parsed.utcoffset().total_seconds() == 0
    # Second resolution: the helper drops microseconds so the value is stable
    # enough to compare between two runs.
    assert parsed.microsecond == 0
    assert stamp == file_timestamp(good_file)


def test_open_context_closes_the_handle_even_when_the_body_raises(
    source: NetCDFSource, tmp_path: Path
) -> None:
    """A failing caller must not leak the handle either."""
    path = write_netcdf(make_dataset(), tmp_path / "boom.nc")
    with pytest.raises(RuntimeError, match="caller exploded"):
        with source.open_context(path):
            raise RuntimeError("caller exploded")

    path.unlink()
    assert not path.exists()