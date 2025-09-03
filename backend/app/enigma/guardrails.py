from typing import List, Tuple, Dict
from itertools import combinations

def hamming_distance(a: List[int], b: List[int]) -> int:
    sa, sb = set(a), set(b)
    return len(sa.symmetric_difference(sb))

def violates_min_hamming(candidate: List[int], selected: List[List[int]], dmin: int) -> bool:
    for g in selected:
        if hamming_distance(candidate, g) < dmin:
            return True
    return False

def build_pair_counts(selected: List[List[int]]) -> Dict[Tuple[int,int], int]:
    counts: Dict[Tuple[int,int], int] = {}
    for g in selected:
        for i,j in combinations(sorted(g), 2):
            counts[(i,j)] = counts.get((i,j), 0) + 1
    return counts

def violates_pair_exposure(candidate: List[int], pair_counts: Dict[Tuple[int,int], int], cap: int) -> bool:
    for i,j in combinations(sorted(candidate), 2):
        if pair_counts.get((i,j), 0) + 1 > cap:
            return True
    return False

def update_pair_counts(candidate: List[int], pair_counts: Dict[Tuple[int,int], int]):
    for i,j in combinations(sorted(candidate), 2):
        pair_counts[(i,j)] = pair_counts.get((i,j), 0) + 1
