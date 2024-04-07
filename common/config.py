# config.py
import logging

logging.basicConfig(filename='pv_simulation.log', level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger()

# Constantes físicas
CHARGE = 1.60217646e-19  # Carga elemental (Coulombs)
BOLTZMANN_CONST = 1.3806503e-23  # Constante de Boltzmann (J/K)
BANDGAP_ENERGY = 1.1  # Energía de banda prohibida para el silicio a 300K (eV)

# Parámetros del modelo PV (Ejemplos)
NOMINAL_TEMP = 298  # Temperatura nominal en Kelvin
IDEALITY_FACTOR = 1.0  # Factor de idealidad
