# ENIGMA API — MVP (Tail Optimizer + Guard-rails)

Este pacote entrega o primeiro corte do **módulo XI (Otimizador de Cauda por Poisson-Binomial)** e do **módulo XVI (Guard-rails de Anticoncentração)**, com uma API FastAPI `/api/generate`.

## O que está implementado

- **/api/generate**: gera K bilhetes (1..100), cada um com 15 números (1..25) otimizados para **maximizar a probabilidade de que o portfólio tenha pelo menos um bilhete com ≥ k acertos** (k configurável, default 13).
- **Tail Optimizer (XI)**: usa **Poisson-Binomial** para calcular `q_k(b) = P(hits(b) ≥ k)` de cada bilhete `b` dado um vetor de marginais `p` (probabilidade de cada número sair no próximo sorteio). Seleção gulosa maximiza `1 - ∏(1 - q_b)`.
- **Guard-rails (XVI)**:
  - `min_hamming`: distância mínima entre qualquer par de bilhetes (default 4).
  - `max_pair_exposure`: teto de exposição por par de números ao longo do portfólio (default 12).

> **Snapshot de probabilidades**: o serviço lê `25` marginais em `sample/p_snapshot.json`. Por padrão o arquivo contém `0.6` para todos (soma ≈ 15). Em produção, gere este arquivo diariamente a partir do seu pipeline (EWMA, FDR etc.).

## Rodando localmente

```bash
cd backend
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload
```

Abra: `http://127.0.0.1:8000/docs`

## Exemplo de requisição

```bash
curl -X POST http://127.0.0.1:8000/api/generate   -H "Content-Type: application/json"   -d '{
    "quantity": 10,
    "mode": "AUTO",
    "objective": "TAIL",
    "tail_target": 13,
    "advanced": {
      "min_hamming": 4,
      "max_pair_exposure": 12,
      "seed": 42
    }
  }'
```

## Estrutura

```
backend/
  app/
    enigma/
      tail_optimizer.py      # Poisson-Binomial (P[S>=k])
      guardrails.py          # Hamming e exposição de pares
    main.py                  # FastAPI + seleção gulosa
    schemas.py               # Pydantic models
  requirements.txt
sample/
  p_snapshot.json            # 25 probabilidades (soma ~15)
```

## Como substituir o snapshot `p` (produção)

1. Gere `p` diariamente (25 números) com o seu pipeline (EWMA, correções, normalização para somar ≈ 15).
2. Escreva o arquivo JSON (lista de 25 floats) e monte no contêiner como `/app/sample/p_snapshot.json` ou ajuste `ENIGMA_P_SNAPSHOT`.

## Notas de design

- Seleção gulosa **por cauda** foca picos (≥13/≥14/≥15) e usa guard-rails para evitar clones e concentração invisível.
- O cálculo Poisson-Binomial usa DP O(n*k), com `n=15` (tamanho do bilhete), rápido o suficiente para K até 100 no MVP.
- As rotas `MEDIA`, `AUTO`, `PARETO` estão encaminhadas para o objetivo TAIL como placeholder. Próximos incrementos: média/pareto, calibração conformal, stress test e Thompson AUTO.

## Variáveis de ambiente

- `ENIGMA_P_SNAPSHOT`: caminho do JSON com as 25 marginais (default: `/app/sample/p_snapshot.json`).

## Licença

Uso interno — Projeto ENIGMA.


## Novidades v1.2
- **Objetivo MEAN** (consistência): maximiza **acertos esperados** por bilhete (∑ p_i do bilhete), com **penalidade opcional de Risk Parity** (exposição agregada por número).
- **Objetivo PARETO**: entrega **3 perfis** — `consistency` (MEAN), `balanced` (merge) e `aggressive` (TAIL). A resposta traz `profiles[]` com jogos e metas por perfil.
- **Risk Parity (simples)**: quando `advanced.risk_parity=true`, o algoritmo penaliza candidatos que concentram números já muito expostos no portfólio.

### Exemplo (PARETO)
```bash
curl -X POST http://127.0.0.1:8000/api/generate   -H "Content-Type: application/json"   -d '{
    "quantity": 20,
    "objective": "PARETO",
    "tail_target": 13,
    "advanced": { "min_hamming": 4, "max_pair_exposure": 12, "risk_parity": true, "seed": 7 }
  }'
```
A resposta inclui `games` (perfil `balanced`) e `profiles` com os três conjuntos.


## Novidades v1.3
- **Calibração (Isotônica)**: ative com `advanced.conformal=true` para reportar `p_success_ge_k_calibrated` (curva é treinada via `/api/feedback`).
- **Teste de Estresse (drift de p)**: ative com `advanced.stress_test=true` e ajuste `stress_sigma` / `stress_samples`. A resposta inclui `robust_min_ge_k`, `robust_p10_ge_k`, `robust_mean_ge_k`.
- **AUTO (Thompson, infra pronta)**: o modo AUTO escolhe objetivo entre MEAN/TAIL13/TAIL14 com base em amostras Beta (persistidas em `/app/state/strategy_posteriors.json`). Atualização de recompensa deve ser enviada futuramente via `/api/feedback` após observação real.

### Endpoints novos
- `POST /api/feedback` — envia pares (k, predicted, outcome) para treinar a calibração isotônica.
- `GET /api/calibration/status` — mostra quantos pontos de calibração existem por k.

### Exemplo (TAIL + calibração + stress)
```bash
curl -X POST http://127.0.0.1:8000/api/generate   -H "Content-Type: application/json"   -d '{
    "quantity": 30,
    "mode": "AUTO",
    "objective": "TAIL",
    "tail_target": 13,
    "advanced": { "conformal": true, "stress_test": true, "stress_sigma": 0.03, "stress_samples": 300, "seed": 11 }
  }'
```
