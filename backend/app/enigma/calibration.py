import os
import json
import bisect
from typing import List, Dict
from .database import get_db_connection

def add_points(k: int, preds: List[float], outcomes: List[int]):
    """Adiciona novos pontos de calibração ao DB e retreina o modelo isotônico para um dado k."""
    assert len(preds) == len(outcomes)
    
    conn = get_db_connection()
    cursor = conn.cursor()
    
    # Insere os novos pontos de forma eficiente.
    points_to_insert = list(zip([k]*len(preds), preds, outcomes))
    cursor.executemany(
        "INSERT INTO calibration_points (k, prediction, outcome) VALUES (?, ?, ?)",
        points_to_insert
    )
    
    # Busca todos os pontos para o k especificado para retreinar o modelo.
    cursor.execute("SELECT prediction, outcome FROM calibration_points WHERE k = ?", (k,))
    rows = cursor.fetchall()
    
    # Limita a quantidade de pontos em memória para evitar uso excessivo.
    if len(rows) > 50000:
        rows = rows[-50000:]

    xs = [row['prediction'] for row in rows]
    ys = [row['outcome'] for row in rows]
    
    # Treina o novo modelo com todos os dados históricos.
    model = fit_isotonic(xs, ys)
    
    # Salva/Atualiza o modelo treinado no banco de dados.
    if model:
        model_json = json.dumps(model)
        cursor.execute(
            "INSERT OR REPLACE INTO calibration_models (k, model_json) VALUES (?, ?)",
            (k, model_json)
        )
    
    conn.commit()
    conn.close()

def fit_isotonic(xs: List[float], ys: List[int]):
    """
    Implementação do algoritmo PAVA (Pool Adjacent Violators Algorithm) para Regressão Isotônica.
    Esta função permanece inalterada.
    """
    if len(xs) < 20:
        return None
    pairs = sorted(zip(xs, ys), key=lambda t: t[0])
    x_sorted = [p[0] for p in pairs]
    y_sorted = [p[1] for p in pairs]
    n = len(x_sorted)
    g = [float(y) for y in y_sorted]
    w = [1.0]*n
    i = 0
    while i < n-1:
        if g[i] <= g[i+1]:
            i += 1
            continue
        new_g = (g[i]*w[i] + g[i+1]*w[i+1]) / (w[i] + w[i+1])
        new_w = w[i] + w[i+1]
        g[i] = new_g; w[i] = new_w
        del g[i+1]; del w[i+1]; del x_sorted[i+1]; del y_sorted[i+1]
        n -= 1
        if i > 0:
            i -= 1
    model = {"x": x_sorted, "g": g}
    return model

def calibrate(k: int, p: float) -> float:
    """Aplica o modelo de calibração isotônica a uma probabilidade p, buscando o modelo no DB."""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT model_json FROM calibration_models WHERE k = ?", (k,))
    row = cursor.fetchone()
    conn.close()

    if not row or not row['model_json']:
        return float(p)
    
    model = json.loads(row['model_json'])
    xs = model.get("x", [])
    gs = model.get("g", [])
    
    if not xs:
        return float(p)
    
    idx = bisect.bisect_right(xs, p) - 1
    if idx < 0:
        return float(gs[0])
    if idx >= len(gs):
        return float(gs[-1])
    return float(gs[idx])

def status_counts() -> Dict[str, int]:
    """Retorna a contagem de pontos de calibração por k, consultando o DB."""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT k, COUNT(id) FROM calibration_points GROUP BY k")
    rows = cursor.fetchall()
    conn.close()
    
    return {str(row['k']): row['COUNT(id)'] for row in rows}