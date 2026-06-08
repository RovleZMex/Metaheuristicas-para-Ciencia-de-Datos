# -*- coding: utf-8 -*-
"""
Motores de inferencia difusa (Mamdani, Tsukamoto, Takagi-Sugeno-Kang) y la
configuración del sistema de **riesgo crediticio** sobre el German Credit Dataset.

Implementación propia con numpy, en el mismo estilo de la libreta 05 del curso
(funciones de membresía trapezoidales, reglas como diccionarios, defuzzificación
por centroide en Mamdani, conjuntos monótonos en Tsukamoto y consecuentes
lineales en TSK). Sin dependencias de librerías difusas externas.

Compartido por las tareas 05 (sistema de inferencia difusa) y 06 (sistema
multiagente difuso).
"""
import numpy as np


# ===========================================================================
# Funciones de membresía y variable lingüística
# ===========================================================================
def trapmf(x, a, b, c, d):
    """Función de membresía trapezoidal definida por (a, b, c, d).

    Casos especiales: a == b produce un hombro izquierdo (membresía 1 hacia la
    izquierda); c == d produce un hombro derecho. Una triangular es el caso
    b == c. Funciona con escalares y con arreglos de numpy.
    """
    x = np.asarray(x, dtype=float)
    up = np.divide(x - a, b - a, out=np.ones_like(x), where=(b > a))
    down = np.divide(d - x, d - c, out=np.ones_like(x), where=(d > c))
    m = np.minimum(np.minimum(up, 1.0), down)
    return np.clip(m, 0.0, 1.0)


class FuzzyVariable:
    """Variable lingüística: un nombre, sus conjuntos difusos y (opcional) su
    universo de discurso. `sets` es un dict {etiqueta: [a, b, c, d]}."""

    def __init__(self, name, sets, universe=None):
        self.name = name
        self.sets = sets
        self.universe = universe

    def fuzzify(self, x):
        """Devuelve {etiqueta: grado de pertenencia} para el valor x."""
        return {lbl: float(trapmf(x, *p)) for lbl, p in self.sets.items()}


def _firing(mu, rule):
    """Grado de activación de una regla: T-norma mínimo sobre sus antecedentes."""
    return min(mu[v][lbl] for v, lbl in rule["antecedentes"].items())


# ===========================================================================
# Sistema de Mamdani
# ===========================================================================
class MamdaniFIS:
    """Inferencia de Mamdani: recorte (clipping) del consecuente por el grado de
    activación, agregación por máximo y defuzzificación por centroide."""

    def __init__(self, input_vars, output_var, rules, n_out=400, default=50.0):
        self.iv = input_vars
        self.ov = output_var
        self.rules = rules
        self.y = np.linspace(output_var.universe[0], output_var.universe[1], n_out)
        self.default = default

    def infer(self, inputs):
        mu = {v: self.iv[v].fuzzify(inputs[v]) for v in self.iv}
        agg = np.zeros_like(self.y)
        for r in self.rules:
            a = _firing(mu, r)
            if a <= 0:
                continue
            curve = np.minimum(a, trapmf(self.y, *self.ov.sets[r["consecuente"]]))
            agg = np.maximum(agg, curve)              # agregación por máximo
        s = agg.sum()
        return self.default if s == 0 else float(np.sum(self.y * agg) / s)


# ===========================================================================
# Sistema de Tsukamoto
# ===========================================================================
class TsukamotoFIS:
    """Inferencia de Tsukamoto: cada consecuente es un conjunto monótono; para
    cada regla se obtiene el valor z cuya membresía iguala al grado de
    activación, y la salida es el promedio ponderado de esos z."""

    def __init__(self, input_vars, output_var, rules, default=50.0):
        self.iv = input_vars
        self.rules = rules
        self.default = default
        self.mono = self._build_monotonic(output_var.sets)

    @staticmethod
    def _build_monotonic(sets):
        """Convierte los conjuntos de salida en versiones monótonas: el primero
        decreciente (hombro izquierdo) y el resto crecientes."""
        labels = list(sets)
        mono = {}
        for i, lbl in enumerate(labels):
            a, b, c, d = sets[lbl]
            mono[lbl] = [a, a, c, d] if i == 0 else [a, b, d, d]
        return mono

    @staticmethod
    def _inverse(alpha, params):
        """Inversa de una membresía monótona: devuelve z tal que mu(z) = alpha."""
        a, b, c, d = params
        if alpha <= 0:
            return a
        if alpha >= 1:
            return d
        if b > a:                      # rama creciente
            return a + alpha * (b - a)
        if d > c:                      # rama decreciente
            return d - alpha * (d - c)
        return (a + d) / 2

    def infer(self, inputs):
        mu = {v: self.iv[v].fuzzify(inputs[v]) for v in self.iv}
        num = den = 0.0
        for r in self.rules:
            a = _firing(mu, r)
            if a <= 0:
                continue
            z = self._inverse(a, self.mono[r["consecuente"]])
            num += a * z
            den += a
        return self.default if den == 0 else num / den


# ===========================================================================
# Sistema de Takagi-Sugeno-Kang (TSK) de primer orden
# ===========================================================================
class TSKFIS:
    """Inferencia TSK de primer orden: cada consecuente es un modelo lineal de
    las entradas (normalizadas a [0, 1]); la salida es el promedio ponderado de
    los modelos locales por su grado de activación."""

    def __init__(self, input_vars, consequents, rules, ranges, default=50.0):
        self.iv = input_vars
        self.cons = consequents
        self.rules = rules
        self.ranges = ranges
        self.default = default

    def infer(self, inputs):
        mu = {v: self.iv[v].fuzzify(inputs[v]) for v in self.iv}
        xn = {v: (inputs[v] - self.ranges[v][0]) / (self.ranges[v][1] - self.ranges[v][0])
              for v in self.iv}
        num = den = 0.0
        for r in self.rules:
            a = _firing(mu, r)
            if a <= 0:
                continue
            m = self.cons[r["consecuente"]]
            y = (m["c"] + m["duration"] * xn["duration"]
                 + m["credit_amount"] * xn["credit_amount"] + m["age"] * xn["age"])
            y = float(np.clip(y, 0, 100))
            num += a * y
            den += a
        return self.default if den == 0 else num / den


# ===========================================================================
# Configuración del problema: riesgo crediticio (German Credit Dataset)
# ===========================================================================
# Rangos observados en el dataset (para normalizar las entradas del TSK).
RANGES = {"duration": (4, 72), "credit_amount": (250, 18424), "age": (19, 75)}

# Conjuntos difusos de entrada (etiquetas lingüísticas).
INPUT_SETS = {
    "duration": {                      # duración del crédito (meses)
        "corta": [0, 0, 8, 18],
        "media": [12, 24, 36, 48],
        "larga": [36, 48, 72, 72]},
    "credit_amount": {                 # monto del préstamo (DM)
        "bajo": [0, 0, 1500, 4000],
        "medio": [2500, 4500, 7000, 10000],
        "alto": [7000, 11000, 18424, 18424]},
    "age": {                           # edad (años)
        "joven": [18, 18, 25, 33],
        "adulto": [27, 35, 48, 58],
        "mayor": [50, 60, 75, 75]},
}

# Conjunto difuso de salida: riesgo crediticio en [0, 100].
RISK_SET = {
    "bajo": [0, 0, 20, 40],
    "medio": [25, 45, 55, 75],
    "alto": [60, 80, 100, 100],
}

# Base de reglas (conocimiento experto de riesgo crediticio).
RULES = [
    {"id": "R1", "antecedentes": {"duration": "corta", "credit_amount": "bajo"}, "consecuente": "bajo"},
    {"id": "R2", "antecedentes": {"duration": "larga", "credit_amount": "alto"}, "consecuente": "alto"},
    {"id": "R3", "antecedentes": {"age": "joven", "credit_amount": "alto"}, "consecuente": "alto"},
    {"id": "R4", "antecedentes": {"duration": "media", "credit_amount": "medio"}, "consecuente": "medio"},
    {"id": "R5", "antecedentes": {"age": "mayor", "duration": "corta"}, "consecuente": "bajo"},
    {"id": "R6", "antecedentes": {"duration": "larga", "age": "joven"}, "consecuente": "alto"},
    {"id": "R7", "antecedentes": {"credit_amount": "bajo", "age": "adulto"}, "consecuente": "bajo"},
]

# Consecuentes lineales del TSK (sobre entradas normalizadas a [0, 1]): el riesgo
# crece con la duración y el monto, y decrece con la edad.
TSK_CONSEQUENTS = {
    "bajo":  {"c": 15, "duration": 10, "credit_amount": 10, "age": -10},
    "medio": {"c": 45, "duration": 10, "credit_amount": 10, "age": -8},
    "alto":  {"c": 70, "duration": 12, "credit_amount": 12, "age": -6},
}

FEATURES = ["duration", "credit_amount", "age"]
THRESHOLD = 50.0    # riesgo > THRESHOLD  =>  se clasifica como "bad" (alto riesgo)


def build_credit_systems():
    """Construye los tres FIS configurados para riesgo crediticio.

    Devuelve un dict {"Mamdani": ..., "Tsukamoto": ..., "TSK": ...}.
    """
    iv = {n: FuzzyVariable(n, s) for n, s in INPUT_SETS.items()}
    ov = FuzzyVariable("riesgo", RISK_SET, universe=(0, 100))
    return {
        "Mamdani": MamdaniFIS(iv, ov, RULES),
        "Tsukamoto": TsukamotoFIS(iv, ov, RULES),
        "TSK": TSKFIS(iv, TSK_CONSEQUENTS, RULES, RANGES),
    }


def input_variables():
    """Devuelve el dict de variables lingüísticas de entrada (para graficar)."""
    return {n: FuzzyVariable(n, s) for n, s in INPUT_SETS.items()}


def output_variable():
    """Devuelve la variable lingüística de salida 'riesgo'."""
    return FuzzyVariable("riesgo", RISK_SET, universe=(0, 100))
