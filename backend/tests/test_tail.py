from app.enigma.tail_optimizer import poisson_binomial_tail

def test_tail_simple():
    # 3 Bernoullis with p=0.5, P(S>=2) = 0.5
    probs = [0.5,0.5,0.5]
    v = poisson_binomial_tail(probs, 2)
    assert abs(v - 0.5) < 1e-6
