"""Removed in iohinspector 0.1.0 with the old data model; see CHANGELOG.md."""

from ._upgrade import removed_names

__getattr__ = removed_names(__name__)
