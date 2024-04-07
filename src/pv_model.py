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
                print(f"Error de convergencia para V={v} V: {e}")
                current_values.append(np.nan)

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
                print(f"Simulación para T={T}°C, G={G}W/m^2 completada.")
        self.df = pd.concat(results_list, ignore_index=True)

    def save_results(self, filename='resultados_simulacion.parquet'):
        """
        Guarda los resultados de la simulación en un archivo parquet.
        :param filename: Nombre del archivo a guardar.
        :return: None
        """
        self.df.to_parquet(filename, index=False)
        print(f"Resultados guardados en '{filename}'.")

    @staticmethod
    def read_results(filename='resultados_simulacion.parquet'):
        """
        Lee los resultados de la simulación desde un archivo parquet.
        :param filename: Nombre del archivo a leer.
        :return: DataFrame con los resultados de la simulación.
        """
        df = pd.read_parquet(filename)
        print(df.info())
        print()
        print(df.sample(5))
        print()
        # Imprimir la el punto de máxima potencia para cada combinación de temperatura e irradiancia
        print(df.loc[df.groupby(['Temperatura', 'Irradiancia'])['Power (W)'].idxmax()])

        return df

    def generate_graphs(self, image_path='../images'):
        G_values = [300, 500, 700, 1000]
        T_values = [25, 35, 45, 55]

        fig, axs = plt.subplots(2, 2, figsize=(10, 8))

        # Gráficas I-V y P-V para diferentes valores de G a 25° C
        for G in G_values:
            resultados, V_max, I_max, P_max = self.pv_model_updated(G, 298)
            axs[0, 0].plot(resultados['Voltaje (V)'], resultados['Corriente (A)'], label=f'G={G} W/m²')
            axs[0, 1].plot(resultados['Voltaje (V)'], resultados['Potencia (W)'], label=f'G={G} W/m²')

        # Gráficas I-V y P-V para diferentes valores de T a G=1000
        for T in T_values:
            resultados, V_max, I_max, P_max = self.pv_model_updated(1000, T + 273.15)
            axs[1, 0].plot(resultados['Voltaje (V)'], resultados['Corriente (A)'], label=f'T={T} °C')
            axs[1, 1].plot(resultados['Voltaje (V)'], resultados['Potencia (W)'], label=f'T={T} °C')

        axs[0, 0].set_xlabel('Voltaje (V)')
        axs[0, 0].set_ylabel('Corriente (A)')
        axs[0, 0].set_title('Curva I-V a 25°C')
        axs[0, 0].legend()
        axs[0, 0].grid(True)
        axs[0, 0].set_ylim(bottom=0)

        axs[0, 1].set_xlabel('Voltaje (V)')
        axs[0, 1].set_ylabel('Potencia (W)')
        axs[0, 1].set_title('Curva P-V a 25°C')
        axs[0, 1].legend()
        axs[0, 1].grid(True)
        axs[0, 1].set_ylim(bottom=0)

        axs[1, 0].set_xlabel('Voltaje (V)')
        axs[1, 0].set_ylabel('Corriente (A)')
        axs[1, 0].set_title('Curva I-V a G=1000 W/m²')
        axs[1, 0].legend()
        axs[1, 0].grid(True)
        axs[1, 0].set_ylim(bottom=0)

        axs[1, 1].set_xlabel('Voltaje (V)')
        axs[1, 1].set_ylabel('Potencia (W)')
        axs[1, 1].set_title('Curva P-V a G=1000 W/m²')
        axs[1, 1].legend()
        axs[1, 1].grid(True)
        axs[1, 1].set_ylim(bottom=0)

        # Crear la carpeta si no existe
        if not os.path.exists(image_path):
            os.makedirs(image_path)

        # Guardar la figura
        plt.tight_layout()
        plt.savefig(os.path.join(image_path, 'curvas_pv.png'), dpi=300)

    def single_graph(self, G, T, image_path='../images'):
        resultados, V_max, I_max, P_max = self.pv_model_updated(G, T)

        # Gráficos
        fig, ax1 = plt.subplots()

        color = 'tab:blue'
        ax1.set_xlabel('Voltaje (V)')
        ax1.set_ylabel('Corriente (A)', color=color)
        ax1.plot(resultados['Voltaje (V)'], resultados['Corriente (A)'], color=color)
        ax1.plot(V_max, I_max, 'ro')
        ax1.axvline(x=V_max, color='gray', linestyle='--')  # Línea punteada
        ax1.tick_params(axis='y', labelcolor=color)
        ax1.grid()

        ax2 = ax1.twinx()
        color = 'tab:green'
        ax2.set_ylabel('Potencia (W)', color=color)
        ax2.plot(resultados['Voltaje (V)'], resultados['Potencia (W)'], color=color)
        ax2.plot(V_max, P_max, 'ro')
        ax2.tick_params(axis='y', labelcolor=color)

        plt.title('Curvas I-V y P-V')

        # Crear tabla debajo de la gráfica
        data = [['V_max (V)', f'{V_max:.2f} V'],
                ['I_max (A)', f'{I_max:.2f} A'],
                ['P_max (W)', f'{P_max:.2f} W']]
        table = plt.table(cellText=data, loc='bottom', colWidths=[0.2, 0.2])

        # Ajustar la posición de la gráfica para hacer espacio para la tabla
        plt.subplots_adjust(bottom=0.2)

        # Crear la carpeta si no existe
        if not os.path.exists(image_path):
            os.makedirs(image_path)

        # Guardar la figura
        plt.tight_layout()
        plt.savefig(os.path.join(image_path, 'curvas_pv_single.png'), dpi=300)

