import numpy as np
import pandas as pd
import glob
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, NullFormatter
import seaborn as sns
from scipy.optimize import fsolve

# Función para abrir y concatenar archivos CSV
def open_csv_conc(pattern, exclude_pattern):
    csv_files = glob.glob(pattern)
    csv_files = [file for file in csv_files if exclude_pattern not in file]
    dfs = []

    for file in csv_files:
        df = pd.read_csv(file)
        dfs.append(df)

    combined_df = pd.concat(dfs, ignore_index=True)
    return combined_df

# Función para encontrar la intersección entre dos líneas
def find_intersection(m, y, m1, y1, x0):
    return fsolve(lambda x: (m*x + y) - (m1*x + y1), x0)[0]

# Función para generar los valores de x según la dirección
def generate_x_values(start, end, length=100):
    return np.linspace(start, end, length)

# Función para agregar líneas de corte personalizadas
def add_custom_cut_lines(ax, lines, color):
    line_style = {'linestyle': '-.', 'linewidth': 3, 'alpha': 0.8, 'zorder': 30}
    outline_style = {'linewidth': 5, 'alpha': 0.4, 'zorder': 29, 'color': 'black'}
    
    if len(lines) == 2:
        # Caso de dos líneas de corte
        (m1, y1, dir1), (m2, y2, dir2) = lines
        x_intersect = find_intersection(m1, y1, m2, y2, 0.0)
        print(f"Intersección encontrada en x={x_intersect} para líneas con pendientes {m1} y {m2}")

        # Definir los rangos de x para cada línea según la dirección
        x_values1 = generate_x_values(x_intersect - 100, x_intersect) if "left" in dir1 else generate_x_values(x_intersect, x_intersect + 100)
        x_values2 = generate_x_values(x_intersect - 100, x_intersect) if "left" in dir2 else generate_x_values(x_intersect, x_intersect + 100)

        y_values1 = m1 * x_values1 + y1
        y_values2 = m2 * x_values2 + y2

        ax.plot(x_values1, y_values1, color=color, **line_style)
        ax.plot(x_values1, y_values1, **outline_style)
        ax.plot(x_values2, y_values2, color=color, **line_style)
        ax.plot(x_values2, y_values2, **outline_style)
    elif len(lines) == 3:
        # Caso de tres líneas de corte
        (m1, y1, dir1), (m2, y2), (m3, y3, dir3) = lines

        # Encontrar intersecciones
        x_intersect_12 = find_intersection(m1, y1, m2, y2, 0.0)
        x_intersect_23 = find_intersection(m2, y2, m3, y3, 0.0)

        print(f"Intersecciones encontradas en x={x_intersect_12}, x={x_intersect_23}")

        # Genera valores de x para las líneas según las intersecciones y direcciones
        x_values1 = generate_x_values(x_intersect_12 - 100, x_intersect_12) if "left" in dir1 else generate_x_values(x_intersect_12, x_intersect_12 + 100)
        x_values2 = generate_x_values(x_intersect_12, x_intersect_23)
        x_values3 = generate_x_values(x_intersect_23 - 100, x_intersect_23) if "left" in dir3 else generate_x_values(x_intersect_23, x_intersect_23 + 100)

        y_values1 = m1 * x_values1 + y1
        y_values2 = m2 * x_values2 + y2
        y_values3 = m3 * x_values3 + y3

        ax.plot(x_values1, y_values1, color=color, **line_style)
        ax.plot(x_values1, y_values1, **outline_style)
        ax.plot(x_values2, y_values2, color=color, **line_style)
        ax.plot(x_values2, y_values2, **outline_style)
        ax.plot(x_values3, y_values3, color=color, **line_style)
        ax.plot(x_values3, y_values3, **outline_style)
    elif len(lines) == 1:
        # Caso de una sola línea de corte
        (m, y) = lines[0]
        x_values = generate_x_values(-100, 100)  # Define un rango adecuado para x
        y_values = m * x_values + y
        ax.plot(x_values, y_values, color=color, **line_style)
        ax.plot(x_values, y_values, **outline_style)

# Cargar datos
df_splus_wise = open_csv_conc("Class_wise_main_unique/*.csv", "simbad")

print("Number of objects just with SPLUS+WISE colors:", len(df_splus_wise))

# Pares específicos basados en el análisis preliminar
specific_pairs = [
    (('W2mag', 'r_PStotal'), ('J0378_PStotal', 'J0430_PStotal')),
    (('J0395_PStotal', 'J0430_PStotal'), ('W2mag', 'i_PStotal')),
    (('i_PStotal', 'z_PStotal'), ('W2mag', 'z_PStotal')),
    (('u_PStotal', 'J0430_PStotal'), ('W1mag', 'W2mag')),
    (('g_PStotal', 'u_PStotal'), ('W1mag', 'i_PStotal')),
    (('W2mag', 'g_PStotal'), ('g_PStotal', 'J0378_PStotal')),
    (('J0430_PStotal', 'J0515_PStotal'), ('i_PStotal', 'J0861_PStotal')),
    (('W1mag', 'z_PStotal'), ('z_PStotal', 'J0660_PStotal')),
    (('W1mag', 'r_PStotal'), ('r_PStotal', 'J0378_PStotal'))
]

# Líneas de corte personalizadas para cada clase y cada plot
custom_cut_lines = {
    0: {
        0: [(-0.613022, -0.786812, "up-left"), (0.299186, 0.749829, "up-right")],
        1: [(-0.613022, -0.786812, "up-left"), (0.396936, 1.964741), (-0.468396, -1.417259, "up-left")],
        4: [(-0.499464, -2.801993)],
    },
    1: {
        0: [(0.657908, -4.000669, "up-right"), (-2.490052, -1.146136, "down-left")],
        1: [(-8.093505, -0.105229, "down-right"), (0.657908, -4.000669, "up-right")],
    },
    2: {
        0: [(0.657908, -4.000669, "up-right"), (-2.490052, -1.146136, "down-left")],
        1: [(0.657908, -4.000669, "up-right"), (-8.093505, -0.105229, "down-left")],
    },
    6: {
        0: [(0.657908, -4.000669, "up-right"), (-2.490052, -1.146136, "down-left")],
        1: [(0.657908, -4.000669, "up-right"), (-8.093505, -0.105229, "down-left")],
    },
    # Ejemplo con tres líneas de corte
    7: {
        0: [(-0.613022, -0.786812, "up-left"), (0.396936, 1.964741), (-0.468396, -1.417259, "up-left")]
    }
}

# Crear figura con subplots
fig, axes = plt.subplots(3, 3, figsize=(18, 14), dpi=300)
axes = axes.flatten()

# Definir colores basados en el número de etiquetas
num_colors = len(df_splus_wise["Label"].unique())
colors = sns.color_palette("tab10", num_colors)
edge_colors = sns.color_palette("dark", num_colors)

# Plotear cada par de diagramas color-color
for plot_index, ((x1, y1), (x2, y2)) in enumerate(specific_pairs):
    ax = axes[plot_index]

    all_x = []
    all_y = []

    for group, (label, group_data) in enumerate(df_splus_wise.groupby("Label")):
        legend_label = "Noise" if label == -1 else f"Group {label}"
        color = colors[group % num_colors]
        edge_color = edge_colors[group % num_colors]
        marker = 'o'

        x_values = group_data[x1] - group_data[y1]
        y_values = group_data[x2] - group_data[y2]
        all_x.extend(x_values)
        all_y.extend(y_values)

        ax.scatter(
            x_values, y_values, c=[color], s=40, marker=marker, 
            edgecolors=[edge_color], linewidth=0.5, label=legend_label, alpha=0.8
        )

        print(f"Plot index: {plot_index}, Label: {label}")  # Este es el lugar correcto para imprimir las etiquetas

        # Agregar líneas de corte personalizadas
        if plot_index in custom_cut_lines and label in custom_cut_lines[plot_index]:
            lines = custom_cut_lines[plot_index][label]
            print(f"Aplicando líneas personalizadas para plot_index {plot_index}, label {label}: {lines}")
            add_custom_cut_lines(ax, lines, color)

    xlabel = x1.replace('_PStotal', '').replace('W1mag', 'W1').replace('W2mag', 'W2')
    ylabel = y1.replace('_PStotal', '').replace('W1mag', 'W1').replace('W2mag', 'W2')
    ax.set_xlabel(f"{xlabel} - {ylabel}", fontsize=22)

    xlabel = x2.replace('_PStotal', '').replace('W1mag', 'W1').replace('W2mag', 'W2')
    ylabel = y2.replace('_PStotal', '').replace('W1mag', 'W1').replace('W2mag', 'W2')
    ax.set_ylabel(f"{xlabel} - {ylabel}", fontsize=22, labelpad=10)

    ax.tick_params(axis='both', labelsize=22)
    ax.xaxis.set_minor_locator(MultipleLocator(0.5))
    ax.yaxis.set_minor_locator(MultipleLocator(0.5))
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.yaxis.set_minor_formatter(NullFormatter())

    ax.set_aspect('auto')

    # Calcular límites dinámicos con margen
    margin = 0.15
    x_min, x_max = min(all_x) - margin, max(all_x) + margin
    y_min, y_max = min(all_y) - margin, max(all_y) + margin
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)

# Ajustar leyenda fuera de la figura, centrada horizontalmente en la parte inferior
handles, labels = ax.get_legend_handles_labels()
fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, -0.05), fontsize=22, ncol=7)

# Ajustar diseño y guardar el plot
plt.tight_layout(rect=[0, 0.1, 1, 1])  # Ajustar para hacer espacio para la leyenda
plt.subplots_adjust(bottom=0.06)  # Asegurar que haya suficiente espacio en la parte inferior
plt.savefig("Figs/color_color_diagrams_multiple_balanced_ColorCut_complex.pdf", format='pdf', bbox_inches='tight', dpi=300)
plt.close()