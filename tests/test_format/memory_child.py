"""Write a large synthetic dataset and print the peak memory as JSON.

Run by test_memory.py in a separate process, so that the peak resident set size
belongs to the write alone: ``python memory_child.py <dataset-dir> <rows> <runs>``.
"""

import json
import resource
import sys
import time

from synthetic import big_trace_chunks

from iohinspector.format import write_dataset


def main() -> None:
    out, n_rows, n_runs = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
    manifest, runs, chunks = big_trace_chunks(
        n_rows, n_runs, 100_000, "memory-test", seed=1
    )
    start = time.perf_counter()
    written = write_dataset(out, manifest, runs, chunks)  # includes validate_dataset
    seconds = time.perf_counter() - start
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    peak_bytes = peak if sys.platform == "darwin" else peak * 1024
    print(
        json.dumps(
            {
                "counts": written["counts"],
                "seconds": seconds,
                "peak_rss_bytes": peak_bytes,
            }
        )
    )


if __name__ == "__main__":
    main()
