from typing import List
import numpy as np

def poisson_binomial_tail(probs: List[float], k: int) -> float:
    """
    Calcula P(S >= k) onde S = soma de Bernoullis independentes com probabilidades dadas.
    Implementação DP O(n*k) suficiente para n<=15.
    """
    n = len(probs)
    k = max(0, min(k, n))
    # dp[j] = prob de ter exatamente j sucessos após processar i variáveis (in-place)
    dp = np.zeros(n+1, dtype=np.float64)
    dp[0] = 1.0
    for p in probs:
        # update backwards
        dp[1:n+1] = dp[1:n+1] * (1 - p) + dp[0:n] * p
        dp[0] = dp[0] * (1 - p)
    return float(np.sum(dp[k:]))
