import sqlite3
import os
import json

# Define o caminho para o arquivo do banco de dados SQLite.
DB_PATH = os.environ.get("ENIGMA_DB_FILE", "/app/state/enigma.db")

def get_db_connection():
    """Cria e retorna uma conexão com o banco de dados SQLite."""
    # Garante que o diretório de estado exista.
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    # Permite acessar os resultados da consulta por nome da coluna.
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    """Inicializa o banco de dados e cria as tabelas se não existirem."""
    conn = get_db_connection()
    cursor = conn.cursor()

    # Tabela para armazenar cada ponto de dado para calibração.
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS calibration_points (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        k INTEGER NOT NULL,
        prediction REAL NOT NULL,
        outcome INTEGER NOT NULL
    )
    """)

    # Tabela para armazenar o modelo isotônico já treinado e serializado.
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS calibration_models (
        k INTEGER PRIMARY KEY,
        model_json TEXT NOT NULL
    )
    """)

    # Tabela para os posteriors do Thompson Sampling (usado no modo AUTO).
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS strategy_posteriors (
        arm TEXT PRIMARY KEY,
        alpha INTEGER NOT NULL,
        beta INTEGER NOT NULL
    )
    """)

    # Verifica se a tabela de posteriors está vazia para popular com os valores iniciais.
    cursor.execute("SELECT COUNT(*) FROM strategy_posteriors")
    if cursor.fetchone()[0] == 0:
        # Valores padrão para as estratégias iniciais.
        initial_posteriors = {"MEAN": [1,1], "TAIL13": [1,1], "TAIL14": [1,1]}
        
        for arm, params in initial_posteriors.items():
            cursor.execute(
                "INSERT INTO strategy_posteriors (arm, alpha, beta) VALUES (?, ?, ?)",
                (arm, params[0], params[1])
            )
    
    conn.commit()
    conn.close()
