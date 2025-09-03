from pydantic import BaseModel, Field, conint, field_validator
from typing import List, Optional, Literal, Dict

Mode = Literal["MEDIA", "CAUDA", "AUTO"]
Objective = Literal["MEAN", "TAIL", "PARETO"]

class AdvancedOpts(BaseModel):
    fixed_numbers: Optional[List[int]] = Field(default=None, description="Números fixos (1..25) a incluir em todos os bilhetes")
    excluded_numbers: Optional[List[int]] = Field(default=None, description="Números proibidos (1..25)")
    max_overlap: Optional[conint(ge=0, le=14)] = Field(default=None, description="Sobreposição máxima entre bilhetes (0..14)")
    seed: Optional[int] = None
    min_hamming: Optional[conint(ge=0, le=15)] = Field(default=4, description="Distância mínima de Hamming entre bilhetes (guard-rail)")
    max_pair_exposure: Optional[conint(ge=1)] = Field(default=12, description="Exposição máxima por par (guard-rail)")
    risk_parity: Optional[bool] = Field(default=False, description="Ativa penalidade simples de exposição agregada por número")
    conformal: Optional[bool] = Field(default=False, description="Aplica calibração isotônica às probabilidades reportadas")
    stress_test: Optional[bool] = Field(default=False, description="Executa teste de estresse por drift em p")
    stress_sigma: Optional[float] = Field(default=0.03, description="Desvio padrão do ruído de drift em p")
    stress_samples: Optional[int] = Field(default=200, description="Número de cenários de estresse a simular")

    @field_validator('fixed_numbers', 'excluded_numbers')
    @classmethod
    def validate_numbers(cls, v):
        if v is None:
            return v
        bad = [x for x in v if x < 1 or x > 25]
        if bad:
            raise ValueError(f"Números fora do intervalo 1..25: {bad}")
        return sorted(list(dict.fromkeys(v)))

class GenerateRequest(BaseModel):
    quantity: conint(ge=1, le=100) = Field(..., description="Quantidade de bilhetes (1..100)")
    mode: Mode = Field("AUTO")
    objective: Objective = Field("TAIL")
    tail_target: conint(ge=11, le=15) = Field(13, description="k alvo para P(hits>=k) no modo TAIL")
    advanced: Optional[AdvancedOpts] = None

class Game(BaseModel):
    id: int
    numbers: List[int]

# --- VERSÃO CORRETA ---
# ProfileOut definido ANTES de GenerateResponse

class ProfileOut(BaseModel):
    name: str
    games: List[Game]
    meta: Dict[str, float]
    profiles: Optional[List['ProfileOut']] = None

class GenerateResponse(BaseModel):
    run_id: str
    engine_version: str
    dataset_snapshot: str
    seed_used: Optional[int]
    mode_used: Mode
    objective_used: Objective
    k_tail: int
    games: List[Game]
    meta: Dict[str, float]
    profiles: Optional[List[ProfileOut]] = None

# --- FIM DA CORREÇÃO ---

class FeedbackItem(BaseModel):
    k: conint(ge=11, le=15)
    predicted: float
    outcome: conint(ge=0, le=1)

class FeedbackRequest(BaseModel):
    items: List[FeedbackItem]

class CalibrationStatus(BaseModel):
    counts: Dict[str, int]

