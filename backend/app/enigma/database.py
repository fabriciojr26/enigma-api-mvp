import sqlite3
import json
import os
from typing import List, Optional, Dict, Any

# Define o caminho para o ficheiro da base de dados.
# No Render, ele será criado no disco persistente.
DB_PATH = os.path.join(os.environ.get("RENDER_DISK_PATH", "."), "enigma_state.db")

def init_db():
    """Inicializa a base de dados e cria as tabelas se não existirem."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    # Tabela para os modelos de calibração isotónica
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS calibration (
            k INTEGER PRIMARY KEY,
            model_json TEXT NOT NULL
        )
    """)
    # Tabela para os posteriores de Thompson Sampling (braços do MAB)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS thompson_posteriors (
            arm TEXT PRIMARY KEY,
            alpha INTEGER NOT NULL,
            beta INTEGER NOT NULL
        )
    """)
    # Tabela para os dados brutos de calibração (pontos x, y)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS calibration_points (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            k INTEGER NOT NULL,
            x REAL NOT NULL,
            y INTEGER NOT NULL
        )
    """)
    # Inicializa os braços de Thompson se a tabela estiver vazia
    cursor.execute("SELECT COUNT(*) FROM thompson_posteriors")
    if cursor.fetchone()[0] == 0:
        initial_arms = [('MEAN', 1, 1), ('TAIL13', 1, 1), ('TAIL14', 1, 1)]
        cursor.executemany("INSERT INTO thompson_posteriors (arm, alpha, beta) VALUES (?, ?, ?)", initial_arms)
    
    conn.commit()
    conn.close()

def get_calibration_model(k: int) -> Optional[Dict[str, Any]]:
    """Obtém o modelo de calibração para um dado k."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT model_json FROM calibration WHERE k = ?", (k,))
    row = cursor.fetchone()
    conn.close()
    if row:
        return json.loads(row[0])
    return None

def save_calibration_model(k: int, model: Dict[str, Any]):
    """Salva ou atualiza um modelo de calibração."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    model_json = json.dumps(model)
    cursor.execute("INSERT OR REPLACE INTO calibration (k, model_json) VALUES (?, ?)", (k, model_json))
    conn.commit()
    conn.close()

def get_thompson_posteriors() -> Dict[str, List[int]]:
    """Obtém os parâmetros alpha e beta para todos os braços."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT arm, alpha, beta FROM thompson_posteriors")
    rows = cursor.fetchall()
    conn.close()
    return {row[0]: [row[1], row[2]] for row in rows}

def update_thompson_posterior(arm: str, reward: int):
    """Atualiza o posterior de um braço com base numa recompensa (0 ou 1)."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT alpha, beta FROM thompson_posteriors WHERE arm = ?", (arm,))
    row = cursor.fetchone()
    if row:
        alpha, beta = row
        if reward == 1:
            alpha += 1
        else:
            beta += 1
        cursor.execute("UPDATE thompson_posteriors SET alpha = ?, beta = ? WHERE arm = ?", (alpha, beta, arm))
        conn.commit()
    conn.close()

def add_calibration_points(k: int, preds: List[float], outcomes: List[int]):
    """Adiciona novos pontos de dados para a calibração."""
    points = list(zip([k]*len(preds), preds, outcomes))
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.executemany("INSERT INTO calibration_points (k, x, y) VALUES (?, ?, ?)", points)
    conn.commit()
    conn.close()

def get_calibration_points(k: int, limit: int = 50000) -> Dict[str, List]:
    """Obtém os pontos de dados mais recentes para a calibração."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT x, y FROM calibration_points WHERE k = ? ORDER BY id DESC LIMIT ?", (k, limit))
    rows = cursor.fetchall()
    conn.close()
    if not rows:
        return {"x": [], "y": []}
    # Inverte para manter a ordem cronológica
    rows.reverse()
    return {"x": [row[0] for row in rows], "y": [row[1] for row in rows]}

def get_calibration_counts() -> Dict[str, int]:
    """Obtém a contagem de pontos de calibração por k."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT k, COUNT(*) FROM calibration_points GROUP BY k")
    rows = cursor.fetchall()
    conn.close()
    return {str(row[0]): row[1] for row in rows}
