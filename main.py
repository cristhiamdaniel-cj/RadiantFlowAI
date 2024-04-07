from src.pv_model import PVModel


def main():
    # Datos del panel Sunset PX-72 bajo condiciones STD
    irradiance = 1000  # W/m^2
    temperature = 25  # Grados Celsius
    isc = 9.35  # A
    voc = 47.4  # V
    num_cells = 72
    series_resistance = 0.39  # Ohms
    shunt_resistance = 545.82  # Ohms

    # Convertir los coeficientes de temperatura de porcentaje por Kelvin a valor absoluto basado en isc y voc
    temp_coeff_isc = isc * (0.037 / 100)  # A/K
    model = PVModel(irradiance, temperature, isc, voc, num_cells, series_resistance, shunt_resistance, temp_coeff_isc)

    # Ejecutar el modelo
    results, vmpp, impp, p_max = model.pv_model(25, 1000)

    # Verificar los resultados contra los valores técnicos especificados
    print(f"Vmpp esperado: 38.4 V, obtenido: {vmpp} V")
    print(f"Impp esperado: 8.84 A, obtenido: {impp} A")
    print(f"Pmax esperado: 340 W, obtenido: {p_max} W")

    # Llamar las gráficas
    g_values = [300, 500, 700, 1000]
    t_values = [25, 35, 45, 55]
    model.generate_graphs(g_values, t_values)
    model.single_graph(irradiance, temperature)


if __name__ == "__main__":
    main()
