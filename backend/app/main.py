from fastapi import FastAPI
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Optional, Tuple, Dict
import numpy as np
import time, os, json, random, math
from app.schemas import GenerateRequest, GenerateResponse, Game, Mode, Objective, ProfileOut, FeedbackRequest, CalibrationStatus
from app.enigma.tail_optimizer import poisson_binomial_tail
from app.enigma.guardrails import violates_min_hamming, build_pair_counts, violates_pair_exposure, update_pair_counts
# Importações do novo módulo de base de dados
from app.enigma.database import init_db, get_calibration_model, save_calibration_model, get_thompson_posteriors, update_thompson_posterior, get_calibration_counts

ENGINE_VERSION = "enigma-v1.4-db"
SNAPSHOT_FILE = os.environ.get("ENIGMA_P_SNAPSHOT", "/app/sample/p_snapshot.json")

app = FastAPI(title="ENIGMA API", version=ENGINE_VERSION)

# --- CONFIGURAÇÃO DE CORS REFORÇADA ---
# Lista de origens permitidas. Adicionamos o seu site do Netlify.
# O "*" é um "wildcard" que permite testar a partir de qualquer origem.
origins = [
    "http://localhost",
    "http://localhost:8080",
    "http://127.0.0.1:8000",
    "https://celestial-heliotrope-650d98.netlify.app",
    "*" 
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# --- FIM DA CONFIGURAÇÃO DE CORS ---


@app.on_event("startup")
def on_startup():
    init_db() # Garante que a base de dados e as tabelas são criadas ao iniciar

def load_probabilities() -> List[float]:
    try:
        with open(SNAPSHOT_FILE, "r") as f:
            p = json.load(f)
        if not isinstance(p, list) or len(p) != 25:
            raise ValueError("Snapshot inválido")
        return [float(x) for x in p]
    except Exception:
        return [0.6]*25

def calibrate(k: int, p: float) -> float:
    model = get_calibration_model(k)
    if not model:
        return p
    
    xs = model.get("x", [])
    gs = model.get("g", [])

    if not xs:
        return float(p)
    
    import bisect
    idx = bisect.bisect_right(xs, p) - 1
    if idx < 0:
        return float(gs[0])
    if idx >= len(gs):
        return float(gs[-1])
    return float(gs[idx])

# As restantes funções (sample_candidate, greedy_tail_selection, etc.) permanecem as mesmas
# ... (O restante código de main.py não precisa de ser alterado)
def sample_candidate(p: List[float], k: int, fixed: Optional[List[int]] = None, excluded: Optional[List[int]] = None, rng: random.Random = None) -> List[int]:
    rng = rng or random
    fixed = fixed or []
    excluded = set(excluded or [])
    universe = [i for i in range(1,26) if i not in excluded and i not in fixed]
    if len(fixed) > k:
        raise ValueError("Mais números fixos do que o tamanho do bilhete")
    remaining = k - len(fixed)
    weights = np.array([p[i-1] for i in universe], dtype=np.float64)
    if weights.sum() <= 0:
        weights = np.ones_like(weights)
    weights = weights / weights.sum()
    picks = rng.choices(universe, weights=weights, k=max(remaining*3, remaining))
    chosen = list(dict.fromkeys(picks))[:remaining]
    if len(chosen) < remaining:
        rest = [i for i in universe if i not in chosen]
        chosen += rng.sample(rest, remaining - len(chosen))
    result = sorted(list(dict.fromkeys(fixed + chosen)))[:k]
    if len(result) < k:
        rest = [i for i in universe if i not in result]
        result += rng.sample(rest, k - len(result))
        result = sorted(result)
    return result

def expected_hits(game: List[int], p: List[float]) -> float:
    return float(sum(p[i-1] for i in game))

def tail_score_for_game(game: List[int], p: List[float], tail_k: int) -> float:
    probs = [p[i-1] for i in game]
    return poisson_binomial_tail(probs, tail_k)

def greedy_tail_selection(K: int, p: List[float], tail_k: int, fixed: Optional[List[int]], excluded: Optional[List[int]], min_hamming: int, max_pair_exp: int, seed: Optional[int]) -> Tuple[List[List[int]], dict]:
    rng = random.Random(seed or int(time.time()))
    selected: List[List[int]] = []
    pair_counts = {}
    prod_fail = 1.0
    attempts_per_slot = 250
    for _ in range(K):
        best_gain = -1.0
        best_game = None
        best_q = 0.0
        pair_counts = build_pair_counts(selected)
        for _try in range(attempts_per_slot):
            cand = sample_candidate(p, k=15, fixed=fixed, excluded=excluded, rng=rng)
            if violates_min_hamming(cand, selected, min_hamming):
                continue
            if violates_pair_exposure(cand, pair_counts, max_pair_exp):
                continue
            q = tail_score_for_game(cand, p, tail_k)
            gain = prod_fail * q
            if gain > best_gain:
                best_gain, best_game, best_q = gain, cand, q
        if best_game is None:
            if min_hamming > 0:
                min_hamming -= 1
                continue
            best_game = sample_candidate(p, k=15, fixed=fixed, excluded=excluded, rng=rng)
            best_q = tail_score_for_game(best_game, p, tail_k)
            best_gain = prod_fail * best_q
        selected.append(best_game)
        prod_fail *= (1.0 - best_q)
        update_pair_counts(best_game, pair_counts)
    success_portfolio = 1.0 - prod_fail
    mean_hits = float(np.mean([expected_hits(g, p) for g in selected])) if selected else 0.0
    return selected, { "p_success_ge_k": float(success_portfolio), "mean_expected_hits": mean_hits }

def greedy_mean_selection(K: int, p: List[float], fixed: Optional[List[int]], excluded: Optional[List[int]], min_hamming: int, max_pair_exp: int, seed: Optional[int], risk_parity: bool=False) -> Tuple[List[List[int]], dict]:
    rng = random.Random(seed or int(time.time()))
    selected: List[List[int]] = []
    pair_counts = {}
    exposure = [0]*26  # index by number
    attempts_per_slot = 250
    lambda_risk = 0.1 if risk_parity else 0.0
    for _ in range(K):
        best_score = -1e9
        best_game = None
        best_mean = 0.0
        pair_counts = build_pair_counts(selected)
        for _try in range(attempts_per_slot):
            cand = sample_candidate(p, k=15, fixed=fixed, excluded=excluded, rng=rng)
            if violates_min_hamming(cand, selected, min_hamming):
                continue
            if violates_pair_exposure(cand, pair_counts, max_pair_exp):
                continue
            m = expected_hits(cand, p)
            penalty = 0.0
            if risk_parity:
                penalty = sum(exposure[i] for i in cand) / 15.0
            score = m - lambda_risk * penalty
            if score > best_score:
                best_score, best_game, best_mean = score, cand, m
        if best_game is None:
            if min_hamming > 0:
                min_hamming -= 1
                continue
            best_game = sample_candidate(p, k=15, fixed=fixed, excluded=excluded, rng=rng)
            best_mean = expected_hits(best_game, p)
        selected.append(best_game)
        for i in best_game:
            exposure[i] += 1
        update_pair_counts(best_game, pair_counts)
    mean_hits = float(np.mean([expected_hits(g, p) for g in selected])) if selected else 0.0
    return selected, { "mean_expected_hits": mean_hits }

def merge_portfolios(p1: List[List[int]], p2: List[List[int]], K: int, min_hamming: int, max_pair_exp: int) -> List[List[int]]:
    merged: List[List[int]] = []
    pair_counts = {}
    i=j=0
    while len(merged) < K and (i < len(p1) or j < len(p2)):
        for source in [1,2]:
            if len(merged) >= K:
                break
            cand = None
            if source==1 and i < len(p1):
                cand = p1[i]; i += 1
            elif source==2 and j < len(p2):
                cand = p2[j]; j += 1
            if cand is None:
                continue
            if violates_min_hamming(cand, merged, min_hamming): 
                continue
            if violates_pair_exposure(cand, pair_counts, max_pair_exp): 
                continue
            merged.append(cand)
            update_pair_counts(cand, pair_counts)
    return merged[:K]

def stress_perturb(p: List[float], sigma: float, rng: random.Random) -> List[float]:
    # Add Gaussian noise, clip to [0,1], then renormalize to sum~15 while preserving shape
    arr = np.array(p, dtype=np.float64)
    noise = rng.normal(0.0, sigma, size=len(arr))
    arr = np.clip(arr + noise, 0.0, 1.0)
    s = arr.sum()
    if s <= 0:
        return [0.6]*25
    arr = arr * (15.0 / s)
    return [float(x) for x in arr]

def stress_eval_portfolio(games: List[List[int]], p: List[float], tail_k: int, samples: int, sigma: float, seed: Optional[int]) -> Dict[str, float]:
    rng = np.random.default_rng(seed if seed is not None else int(time.time()))
    vals = []
    for _ in range(samples):
        pp = stress_perturb(p, sigma, rng)
        # portfolio success ~= 1 - ∏(1 - q_b)
        prod_fail = 1.0
        for g in games:
            q = poisson_binomial_tail([pp[i-1] for i in g], tail_k)
            prod_fail *= (1.0 - q)
        vals.append(1.0 - prod_fail)
    arr = np.array(vals, dtype=np.float64)
    return {
        "robust_min_ge_k": float(np.min(arr)),
        "robust_p10_ge_k": float(np.percentile(arr, 10)),
        "robust_mean_ge_k": float(np.mean(arr))
    }

def auto_pick_objective() -> Objective:
    post = get_thompson_posteriors()
    draws = {}
    for arm, ab in post.items():
        a,b = ab
        draws[arm] = random.betavariate(a, b)
    arm = max(draws.items(), key=lambda x: x[1])[0]
    if arm == "MEAN":
        return "MEAN"
    if arm == "TAIL14":
        return "TAIL"
    return "TAIL"

@app.post("/api/generate", response_model=GenerateResponse)
def generate(req: GenerateRequest):
    p = load_probabilities()
    adv = req.advanced
    fixed = adv.fixed_numbers if adv else None
    excluded = adv.excluded_numbers if adv else None
    min_hamming = adv.min_hamming if adv else 4
    max_pair_exp = adv.max_pair_exposure if adv else 12
    risk_parity = adv.risk_parity if adv and adv.risk_parity else False
    seed = adv.seed if adv else None
    apply_conformal = adv.conformal if adv else False
    do_stress = adv.stress_test if adv else False
    stress_sigma = adv.stress_sigma if adv else 0.03
    stress_samples = adv.stress_samples if adv else 200

    tail_k = int(req.tail_target)
    K = int(req.quantity)

    objective_used = req.objective
    if req.mode == "AUTO" and objective_used == "PARETO":
        objective_used = auto_pick_objective()

    run_id = f"eng-{int(time.time())}"
    mode_used = req.mode

    def attach_calibration(meta: Dict[str,float]) -> Dict[str,float]:
        out = dict(meta)
        if "p_success_ge_k" in meta and apply_conformal:
            out["p_success_ge_k_calibrated"] = float(calibrate(tail_k, float(meta["p_success_ge_k"])))
        return out

    if objective_used == "MEAN":
        games, meta = greedy_mean_selection(
            K=K, p=p, fixed=fixed, excluded=excluded,
            min_hamming=min_hamming, max_pair_exp=max_pair_exp, seed=seed, risk_parity=risk_parity
        )
        if do_stress:
            meta.update(stress_eval_portfolio(games, p, tail_k, stress_samples, stress_sigma, seed))
        meta = attach_calibration(meta)
        resp = GenerateResponse(
            run_id=run_id,
            engine_version=ENGINE_VERSION,
            dataset_snapshot=os.path.basename(SNAPSHOT_FILE),
            seed_used=seed,
            mode_used=mode_used,
            objective_used=objective_used,
            k_tail=tail_k,
            games=[Game(id=i+1, numbers=g) for i,g in enumerate(games)],
            meta={k: float(v) for k,v in meta.items()}
        )
        return JSONResponse(content=resp.model_dump())

    if objective_used == "PARETO":
        games_mean, meta_mean = greedy_mean_selection(
            K=K, p=p, fixed=fixed, excluded=excluded,
            min_hamming=min_hamming, max_pair_exp=max_pair_exp, seed=seed, risk_parity=risk_parity
        )
        games_tail, meta_tail = greedy_tail_selection(
            K=K, p=p, tail_k=tail_k, fixed=fixed, excluded=excluded,
            min_hamming=min_hamming, max_pair_exp=max_pair_exp, seed=seed
        )
        games_bal = merge_portfolios(games_mean, games_tail, K, min_hamming, max_pair_exp)

        if do_stress:
            meta_tail.update(stress_eval_portfolio(games_tail, p, tail_k, stress_samples, stress_sigma, seed))
            meta_mean.update(stress_eval_portfolio(games_mean, p, tail_k, stress_samples, stress_sigma, seed))
        meta_tail = attach_calibration(meta_tail)
        meta_mean = attach_calibration(meta_mean)
        profiles = [
            ProfileOut(name="consistency", games=[Game(id=i+1, numbers=g) for i,g in enumerate(games_mean)], meta={k: float(v) for k,v in meta_mean.items()}),
            ProfileOut(name="balanced", games=[Game(id=i+1, numbers=g) for i,g in enumerate(games_bal)], meta={"note": 1.0}),
            ProfileOut(name="aggressive", games=[Game(id=i+1, numbers=g) for i,g in enumerate(games_tail)], meta={k: float(v) for k,v in meta_tail.items()})
        ]
        resp = GenerateResponse(
            run_id=run_id,
            engine_version=ENGINE_VERSION,
            dataset_snapshot=os.path.basename(SNAPSHOT_FILE),
            seed_used=seed,
            mode_used=mode_used,
            objective_used="PARETO",
            k_tail=tail_k,
            games=[Game(id=i+1, numbers=g) for i,g in enumerate(games_bal)],
            meta={"profiles": 3},
            profiles=profiles
        )
        return JSONResponse(content=resp.model_dump())

    # Default / TAIL objective
    games, meta = greedy_tail_selection(
        K=K, p=p, tail_k=tail_k, fixed=fixed, excluded=excluded,
        min_hamming=min_hamming, max_pair_exp=max_pair_exp, seed=seed
    )
    if do_stress:
        meta.update(stress_eval_portfolio(games, p, tail_k, stress_samples, stress_sigma, seed))
    meta = attach_calibration(meta)
    resp = GenerateResponse(
        run_id=run_id,
        engine_version=ENGINE_VERSION,
        dataset_snapshot=os.path.basename(SNAPSHOT_FILE),
        seed_used=seed,
        mode_used=mode_used,
        objective_used="TAIL",
        k_tail=tail_k,
        games=[Game(id=i+1, numbers=g) for i,g in enumerate(games)],
        meta={k: float(v) for k,v in meta.items()}
    )
    return JSONResponse(content=resp.model_dump())

@app.post("/api/feedback")
def feedback(req: FeedbackRequest):
    from app.enigma.calibration import add_points as calib_add_points
    ks = {}
    for item in req.items:
        ks.setdefault(item.k, {"preds": [], "outs": []})
        ks[item.k]["preds"].append(float(item.predicted))
        ks[item.k]["outs"].append(int(item.outcome))
    
    objective_map = {13: "TAIL13", 14: "TAIL14"}
    for k, d in ks.items():
        calib_add_points(int(k), d["preds"], d["outs"])
        
        # Atualiza o posterior de Thompson
        arm = objective_map.get(k)
        if arm:
            # Recompensa é 1 se pelo menos um resultado foi 1, senão 0
            reward = 1 if any(out == 1 for out in d["outs"]) else 0
            update_thompson_posterior(arm, reward)

    return {"status": "ok", "updated_ks": list(map(str, ks.keys()))}


@app.get("/api/calibration/status", response_model=CalibrationStatus)
def calibration_status():
    return {"counts": {str(k): v for k,v in get_calibration_counts().items()}}

