import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import fsolve
from common import config


def safe_exp(x):
    """
    Calcula el exponencial de x, evitando overflow en el cálculo.
    :param x: Valor a calcular el exponencial.
    :return: Exponencial de x.
    """
    x_clipped = np.clip(x, None, 700)  # 700 es un valor seguro para evitar overflow en np.exp
    return np.exp(x_clipped)


class PVModel:
    """
    Modelo de un panel fotovoltaico usando el modelo de los 5 parámetros.
    """

    def __init__(self, irradiance, temperature, isc, voc, num_cells, series_resistance, shunt_resistance,
                 temp_coefficient):
        self.irradiance = irradiance  # Irradiancia en W/m^2
        self.temperature = temperature  # Temperatura en grados Celsius
        self.isc = isc  # Corriente de cortocircuito en A
        self.voc = voc  # Tensión de circuito abierto en V
        self.num_cells = num_cells  # Número de celdas en serie
        self.series_resistance = series_resistance  # Resistencia en serie en ohmios
        self.shunt_resistance = shunt_resistance  # Resistencia en paralelo en ohmios
        self.temp_coefficient = temp_coefficient  # Coeficiente de temperatura en A/°C
        self.df = pd.DataFrame()  # DataFrame para almacenar los resultados de la simulación

    def pv_model(self, temperature=None, irradiance=None):
        """
        Simula el comportamiento de un panel fotovoltaico bajo ciertas condiciones de temperatura e irradiancia.
        :param temperature: Temperatura en grados Celsius.
        :param irradiance: Irradiancia en W/m^2.
        :return: DataFrame con los valores de corriente, voltaje y potencia para el panel.
        """
        config.logger.info(f'Iniciando simulación para temperatura={temperature} °C, irradiancia={irradiance} W/m²')
        self.validate_inputs(temperature, irradiance)
        temperature_k = temperature + 273.15  # Convertir a Kelvin

        # Corriente de saturación reversa
        irs = self.isc / (np.exp((config.CHARGE * self.voc) / (
                config.IDEALITY_FACTOR * self.num_cells * config.BOLTZMANN_CONST * config.NOMINAL_TEMP)) - 1)
        # Corriente de saturación en condiciones STC
        i_o = irs * ((temperature_k / config.NOMINAL_TEMP) ** 3) * safe_exp(
            (config.CHARGE * config.BANDGAP_ENERGY / (config.IDEALITY_FACTOR * config.BOLTZMANN_CONST)) * (
                    1 / config.NOMINAL_TEMP - 1 / temperature_k))
        # Foto corriente
        i_ph = (self.isc + self.temp_coefficient * (temperature_k - config.NOMINAL_TEMP)) * (irradiance / 1000)

        def current_voltage_relation(volt, i):
            """
            Relación I-V para el modelo de diodo del panel PV.
            :param volt: Tensión en el punto de operación.
            :param i: Corriente en el punto de operación.
            :return: Diferencia entre la corriente foto generada y la corriente de saturación.
            """
            # Corriente Shunt
            ish = (volt + i * self.series_resistance) / self.shunt_resistance

            return i_ph - i_o * (safe_exp((config.CHARGE * (volt + i * self.series_resistance)) / (
                    config.IDEALITY_FACTOR * config.BOLTZMANN_CONST * self.num_cells * temperature_k)) - 1) - ish - i

        voltage_values = np.linspace(0, self.voc, 100)  # Valores de voltaje para evaluar
        current_values = []

        for v in voltage_values:
            try:
                # Se utiliza un valor inicial cercano a isc y se ajusta la función para que fsolve trabaje correctamente
                i_solution = fsolve(lambda i: current_voltage_relation(v, i), self.isc)[0]
                current_values.append(i_solution)
            except RuntimeError as e:
                config.logger.error(f'Error de convergencia para V={v} V: {e}')
                current_values.append(np.nan)

        config.logger.info(f'Finalizando simulación. Vmpp={voltage_values[np.argmax(current_values)]} V, '
                           f'Impp={np.max(current_values)} A, '
                           f'Pmax={np.max(voltage_values * np.array(current_values))} W')

        power_values = voltage_values * np.array(current_values)
        results = pd.DataFrame(
            {'Voltage (V)': voltage_values, 'Current (A)': current_values, 'Power (W)': power_values})

        max_power_idx = results['Power (W)'].idxmax()
        vmpp = results.iloc[max_power_idx]['Voltage (V)']
        impp = results.iloc[max_power_idx]['Current (A)']
        p_max = results.iloc[max_power_idx]['Power (W)']

        results.to_csv('results.csv', index=False)

        return results, vmpp, impp, p_max

    @staticmethod
    def validate_inputs(temperature=None, irradiance=None):
        """
        Valida que los valores de temperatura e irradiancia estén dentro de los rangos permitidos.
        :param temperature: Temperatura en grados Celsius.
        :param irradiance: Irradiancia en W/m^2.
        :return: None
        """
        if not (0 < irradiance <= 1500):
            raise ValueError("Irradiancia fuera de rango.")
        if not (0 < temperature <= 100):
            raise ValueError("Temperatura fuera de rango.")

    def run_simulation(self, temp_range, irradiance_range):
        """
        Ejecuta la simulación para cada combinación de temperatura e irradiancia en los rangos especificados.
        :param temp_range: Range de temperaturas a simular.
        :param irradiance_range: Range de irradiancias a simular.
        :return: None
        """
        results_list = []
        for T in temp_range:
            for G in irradiance_range:
                resultados, vmpp, impp, p_max = self.pv_model(temperature=T, irradiance=G)
                resultados['Temperatura'] = T
                resultados['Irradiancia'] = G
                results_list.append(resultados)
                config.logger.info(
                    f"Simulación finalizada para T={T} °C, G={G} W/m². Vmpp={vmpp:.2f} V, "
                    f"Impp={impp:.2f} A, Pmax={p_max:.2f} W")
        self.df = pd.concat(results_list, ignore_index=True)

    def save_results(self, filename='resultados_simulacion.parquet'):
        """
        Guarda los resultados de la simulación en un archivo parquet.
        :param filename: Nombre del archivo a guardar.
        :return: None
        """
        self.df.to_parquet(filename, index=False)
        config.logger.info(f"Resultados guardados en {filename}")

    @staticmethod
    def read_results(filename='resultados_simulacion.parquet'):
        """
        Lee los resultados de la simulación desde un archivo parquet.
        :param filename: Nombre del archivo a leer.
        :return: None
        """
        df = pd.read_parquet(filename)
        print(df.info())
        print()
        print(df.sample(5))
        print()
        print(df.describe())
        print()
        df_sub = df[(df['Temperatura'].isin([15, 25, 35, 45, 55])) & (df['Irradiancia'].isin([100, 500, 1000]))]
        print(df_sub.loc[df_sub.groupby(['Temperatura', 'Irradiancia'])['Power (W)'].idxmax()])

    def generate_graphs(self, G_values, T_values, image_path='./img'):

        fig, axs = plt.subplots(2, 2, figsize=(10, 8))

        # Gráficas I-V y P-V para diferentes valores de G a 25° C
        for G in G_values:
            resultados, v_max, i_max, p_max = self.pv_model(25, G)
            axs[0, 0].plot(resultados['Voltage (V)'], resultados['Current (A)'], label=f'G={G} W/m²')
            axs[0, 1].plot(resultados['Voltage (V)'], resultados['Power (W)'], label=f'G={G} W/m²')

        # Gráficas I-V y P-V para diferentes valores de T a G=1000
        for T in T_values:
            resultados, v_max, i_max, p_max = self.pv_model(T, 1000)
            axs[1, 0].plot(resultados['Voltage (V)'], resultados['Current (A)'], label=f'T={T} °C')
            axs[1, 1].plot(resultados['Voltage (V)'], resultados['Power (W)'], label=f'T={T} °C')

        axs[0, 0].set_xlabel('Voltage (V)')
        axs[0, 0].set_ylabel('Current (A)')
        axs[0, 0].set_title('Curva I-V a 25°C')
        axs[0, 0].legend()
        axs[0, 0].grid(True)
        axs[0, 0].set_ylim(bottom=0)

        axs[0, 1].set_xlabel('Voltage (V)')
        axs[0, 1].set_ylabel('Power (W)')
        axs[0, 1].set_title('Curva P-V a 25°C')
        axs[0, 1].legend()
        axs[0, 1].grid(True)
        axs[0, 1].set_ylim(bottom=0)

        axs[1, 0].set_xlabel('Voltage (V)')
        axs[1, 0].set_ylabel('Current (A)')
        axs[1, 0].set_title('Curva I-V a G=1000 W/m²')
        axs[1, 0].legend()
        axs[1, 0].grid(True)
        axs[1, 0].set_ylim(bottom=0)

        axs[1, 1].set_xlabel('Voltage (V)')
        axs[1, 1].set_ylabel('Power (W)')
        axs[1, 1].set_title('Curva P-V a G=1000 W/m²')
        axs[1, 1].legend()
        axs[1, 1].grid(True)
        axs[1, 1].set_ylim(bottom=0)

        # Guardar la figura
        plt.tight_layout()
        plt.savefig(os.path.join(image_path, 'curvas_pv.png'), dpi=300)
        config.logger.info(f"Gráficas guardadas en {os.path.join(image_path, 'curvas_pv.png')}")

    def single_graph(self, G, T, image_path='./img'):
        resultados, v_max, i_max, p_max = self.pv_model(T, G)
        # Gráficos
        fig, ax1 = plt.subplots()

        color = 'tab:blue'
        ax1.set_xlabel('Voltage (V)')
        ax1.set_ylabel('Current (A)', color=color)
        ax1.plot(resultados['Voltage (V)'], resultados['Current (A)'], color=color)
        ax1.plot(v_max, i_max, 'ro')
        ax1.axvline(x=v_max, color='gray', linestyle='--')  # Línea punteada
        ax1.tick_params(axis='y', labelcolor=color)
        ax1.grid()

        ax2 = ax1.twinx()
        color = 'tab:green'
        ax2.set_ylabel('Power (W)', color=color)
        ax2.plot(resultados['Voltage (V)'], resultados['Power (W)'], color=color)
        ax2.plot(v_max, p_max, 'ro')
        ax2.tick_params(axis='y', labelcolor=color)

        plt.title('Curvas I-V y P-V')

        # Crear tabla debajo de la gráfica
        data = [['v_max (V)', f'{v_max:.2f} V'],
                ['i_max (A)', f'{i_max:.2f} A'],
                ['p_max (W)', f'{p_max:.2f} W']]
        plt.table(cellText=data, loc='bottom', colWidths=[0.2, 0.2])

        # Ajustar la posición de la gráfica para hacer espacio para la tabla
        plt.subplots_adjust(bottom=0.2)

        # Guardar la figura
        plt.tight_layout()
        plt.savefig(os.path.join(image_path, 'curvas_pv_single.png'), dpi=300)
        config.logger.info(f"Gráfica guardada en {os.path.join(image_path, 'curvas_pv_single.png')}")
