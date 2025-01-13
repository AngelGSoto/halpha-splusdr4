import numpy as np
import pandas as pd
import glob
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, NullFormatter
import seaborn as sns

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
def find_intersection(m1, b1, m2, b2):
    if m1 == m2:  # No hay intersección si las líneas son paralelas
        return None, None
    x_intersect = (b2 - b1) / (m1 - m2)
    y_intersect = m1 * x_intersect + b1
    return x_intersect, y_intersect

# Función para agregar líneas de corte personalizadas
def add_custom_cut_lines(ax, lines, color):
    for i in range(len(lines) - 1):
        m1, b1 = lines[i]
        m2, b2 = lines[i + 1]
        x_intersect, y_intersect = find_intersection(m1, b1, m2, b2)
        if x_intersect is not None and y_intersect is not None:
            x_limits = ax.get_xlim()
            y_limits = ax.get_ylim()
            x_range1 = np.linspace(x_limits[0], x_intersect, 200)
            y_range1 = m1 * x_range1 + b1
            x_range2 = np.linspace(x_intersect, x_limits[1], 200)
            y_range2 = m2 * x_range2 + b2
            ax.plot(x_range1, y_range1, color=color, linestyle='-', linewidth=2)
            ax.plot(x_range2, y_range2, color=color, linestyle='-', linewidth=2)
            if x_limits[0] <= x_intersect <= x_limits[1] and y_limits[0] <= y_intersect <= y_limits[1]:
                ax.scatter([x_intersect], [y_intersect], color=color, s=50, zorder=5)

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
        0: [(-0.644058, -1.069326), (0.246861, 0.605218)],
        1: [(-0.644058, -1.069326), (0.262123, 1.593165), (-0.663661, -2.423346)],
    },
    1: {
        0: [(-0.644058, -1.069326), (0.246861, 0.605218), (0.679, 0.68998)],
        1: [(0.4, 0.9), (-1.2, 3.2), (-0.5, 1.8)],
    },
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
    points_per_class = []

    for group, (label, group_data) in enumerate(df_splus_wise.groupby("Label")):
        legend_label = "Noise" if label == -1 else f"Group {label}"
        color = colors[group % num_colors]
        edge_color = edge_colors[group % num_colors]
        marker = 'o'

        x_values = group_data[x1] - group_data[y1]
        y_values = group_data[x2] - group_data[y2]
        points = np.column_stack((x_values, y_values))
        points_per_class.append((points, color, label))
        all_x.extend(x_values)
        all_y.extend(y_values)

        ax.scatter(
            x_values, y_values, c=[color], s=90, marker=marker, 
            edgecolors=[edge_color], linewidth=0.5, label=legend_label, alpha=0.7
        )

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
    margin = 0.2
    x_min, x_max = min(all_x) - margin, max(all_x) + margin
    y_min, y_max = min(all_y) - margin, max(all_y) + margin
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)

    # Agregar líneas de corte personalizadas
    for points, color, label in points_per_class:
        if label in custom_cut_lines and plot_index in custom_cut_lines[label]:
            lines = custom_cut_lines[label][plot_index]
            add_custom_cut_lines(ax, lines, color)

    if plot_index == 0:
        fig.legend(loc='lower center', bbox_to_anchor=(0.5, -0.05), fontsize=22, ncol=7)

plt.subplots_adjust(wspace=0.3, hspace=0.3)
plt.tight_layout()
plt.savefig(
    "Figs/color_color_diagrams_multiple_balanced_ColorCut.pdf", 
    format='pdf', bbox_inches='tight', dpi=300
)
plt.close()
