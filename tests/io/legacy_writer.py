"""A writer of legacy IOHprofiler logs for the reader tests (WP I06).

It mimics the IOHexperimenter 0.2 ``Default`` logger: an ``.info`` file and up to
four data files per (function, dimension), each run starting with a header line.
``.dat`` logs improvements, ``.cdat`` every evaluation, ``.idat`` every
``interval``-th evaluation and ``.tdat`` the evaluations 1, 2, 5, 10, 20, 50, ...;
every file also gets the run's last evaluation, as the logger writes it on close.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

HEADER = (
    '"function evaluation" "current f(x)" "best-so-far f(x)" '
    '"current af(x)+b" "best af(x)+b"'
)


@dataclass
class LegacySet:
    """One algorithm on one function: runs are lists of f values, one per evaluation."""

    runs: list[list[float]]
    kinds: tuple[str, ...] = ("dat",)
    function_id: int = 1
    dimension: int = 2
    maximization: bool = False
    interval: int = 3
    params: dict[str, list[list[float]]] = field(default_factory=dict)
    positions: list[list[list[float]]] | None = None
    header_extra: str = ""

    def _rows(self, run: int) -> list[tuple[int, str]]:
        ys = self.runs[run]
        best = ys[0]
        out = []
        for i, y in enumerate(ys):
            improved = i == 0 or (y > best if self.maximization else y < best)
            if improved:
                best = y
            cols = [str(i + 1)] + [f"{v:+.5e}" for v in (y, best, y, best)]
            cols += [f"{p[run][i]:.3f}" for p in self.params.values()]
            if self.positions is not None:
                cols += [f"{x:f}" for x in self.positions[run][i]]
            out.append(
                (i + 1, " ".join(cols)) if improved else (-(i + 1), " ".join(cols))
            )
        return out

    def _keep(self, kind: str, evaluation: int, improved: bool) -> bool:
        if kind == "cdat":
            return True
        if kind == "dat":
            return improved
        if kind == "idat":
            return evaluation % self.interval == 0
        return str(evaluation).strip("0") in ("1", "2", "5")

    def header(self) -> str:
        names = HEADER + "".join(f' "{k}"' for k in self.params)
        if self.positions is not None:
            names += "".join(f' "x{i}"' for i in range(self.dimension))
        return names

    def data_file(self, kind: str) -> str:
        lines = []
        for run in range(len(self.runs)):
            lines.append(self.header())
            rows = self._rows(run)
            for k, (signed, text) in enumerate(rows):
                evaluation, improved = abs(signed), signed > 0
                if self._keep(kind, evaluation, improved) or k == len(rows) - 1:
                    lines.append(text)
        return "\n".join(lines) + "\n"

    def info_line(self) -> str:
        best = []
        for ys in self.runs:
            b = max(ys) if self.maximization else min(ys)
            best.append(f"{len(best) + 1}:{len(ys)}|{b:g}")
        return ", ".join(best)

    def write(self, folder: Path, algorithm: str = "algA") -> Path:
        """Write the files under ``folder``; returns the ``.info`` path."""
        fid, dim = self.function_id, self.dimension
        sub = folder / f"data_f{fid}"
        sub.mkdir(parents=True, exist_ok=True)
        for kind in self.kinds:
            (sub / f"IOHprofiler_f{fid}_DIM{dim}.{kind}").write_text(
                self.data_file(kind)
            )
        info = folder / f"IOHprofiler_f{fid}.info"
        direction = "T" if self.maximization else "F"
        text = (
            f'suite = "PBO", funcId = {fid}, DIM = {dim}, '
            f'maximization = "{direction}", '
            f'algId = "{algorithm}", algInfo = "test"{self.header_extra}\n%\n'
            f"data_f{fid}/IOHprofiler_f{fid}_DIM{dim}.dat, {self.info_line()}\n"
        )
        with open(info, "a", encoding="utf-8") as fh:
            fh.write(text)
        return info
