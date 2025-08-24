# helper to split into rows of 2 or 3 (never a single 1)
from typing import List, Union, Optional

def make_rows(symbols: List[str]) -> List[List[Optional[str]]]:
    n = len(symbols)
    if n == 0:
        return []
    if n == 1:
        # force two columns: [sym, None]
        return [[symbols[0], None]]

    rows: list[list[str|None]] = []
    # if remainder 1, peel off 4 for two rows of 2
    if n % 3 == 1:
        # build rows of 3 from head
        for i in range(0, n - 4, 3):
            rows.append(symbols[i : i + 3])
        # last 4 → two rows of 2
        tail = symbols[n - 4 : n]
        rows.append(tail[0:2])
        rows.append(tail[2:4])
    else:
        # remainder 0 or 2
        main_end = n - (n % 3)
        for i in range(0, main_end, 3):
            rows.append(symbols[i : i + 3])
        if n % 3 == 2:
            rows.append(symbols[main_end : main_end + 2])
    return rows
