import json
import glob
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Define spectral type ranges for MS stars and giants
MS_star_types = ['o5v', 'o9v', 'b0v', 'b1v', 'b3v', 'b57v', 'b8v', 'b9v',
                 'a0v', 'a2v', 'a3v', 'a5v', 'a7v', 'f0v', 'f2v', 'f5v',
                 'f6v', 'f8v', 'g0v', 'g2v', 'g5v', 'g8v', 'k0v', 'k2v',
                 'k3v', 'k4v', 'k5v', 'k7v', 'm0v', 'm1v', 'm2v', 'm2p5v',
                 'm3v', 'm4v', 'm5v', 'm6v']

giant_types = ['b2ii', 'b5ii', 'a0iii', 'a3iii', 'a5iii', 'a7iii', 'f0iii',
               'f2iii', 'f5iii', 'g0iii', 'g5iii', 'g8iii', 'k0iii', 'k1iii',
               'k2iii', 'k3iii', 'k4iii', 'k5iii', 'm0iii', 'm1iii', 'm2iii',
               'm3iii', 'm4iii', 'm5iii', 'm6iii', 'm7iii', 'm8iii', 'm9iii',
               'm10iii']

def filter_mag(e, s, f1, f2, f3, data):
    '''
    Calculate the colors using any set of filters
    '''
    col, col0, prefixes = [], [], []
    if data['id'].endswith(e):
        prefix = data['id'].split("-")[0]
        if prefix.startswith(str(s)):
            filter1 = data[f1]
            filter2 = data[f2]
            filter3 = data[f3]
            diff = filter1 - filter2
            diff0 = filter1 - filter3
            col.append(diff)
            col0.append(diff0)
            prefixes.append(prefix)
    
    return col, col0, prefixes

def classify_star(prefix):
    """
    Classify the star as MS or Giant based on the spectral type prefix
    """
    if prefix in MS_star_types:
        return 'MS'
    elif prefix in giant_types:
        return 'Giant'
    else:
        return None  # Return None if classification fails

def plot_mag(f1, f2, f3):
    MS_A1, MS_B1, giant_A1, giant_B1, MS_prefixes, giant_prefixes = [], [], [], [], [], []
    for file_name in file_list:
        with open(file_name) as f:
            data = json.load(f)
            x, y, prefixes = filter_mag("Star", "", f1, f2, f3, data)
            for i, prefix in enumerate(prefixes):
                classification = classify_star(prefix)
                if classification == 'MS':
                    MS_A1.append(x[i])
                    MS_B1.append(y[i])
                    MS_prefixes.append(prefix)
                elif classification == 'Giant':
                    giant_A1.append(x[i])
                    giant_B1.append(y[i])
                    giant_prefixes.append(prefix)
    
    # Sort the points based on their x-coordinate
    MS_A1, MS_B1, MS_prefixes = zip(*sorted(zip(MS_A1, MS_B1, MS_prefixes), key=lambda x: x[0]))
    giant_A1, giant_B1, giant_prefixes = zip(*sorted(zip(giant_A1, giant_B1, giant_prefixes), key=lambda x: x[0]))
    
    return MS_A1, MS_B1, giant_A1, giant_B1, MS_prefixes, giant_prefixes

# Modify the file pattern to match the JSON files
pattern = "../MS_stars/*.json"
file_list = glob.glob(pattern)

A1_MS, B1_MS, A1_giant, B1_giant, MS_prefixes, giant_prefixes = plot_mag("F0626_rSDSS", "F0660", "F0769_iSDSS")

# Convert lists to numpy arrays
A1_MS = np.array(A1_MS)
B1_MS = np.array(B1_MS)
A1_giant = np.array(A1_giant)
B1_giant = np.array(B1_giant)

# Plotting
with sns.axes_style("ticks"):
    fig, ax = plt.subplots(figsize=(15, 11))
    ax.spines["top"].set_visible(False)  
    ax.spines["right"].set_visible(False)
    plt.xlabel(r"$r - i$", fontsize=35)
    plt.ylabel(r"$r - J0660$", fontsize=35)
    plt.tick_params(axis='x', labelsize=35) 
    plt.tick_params(axis='y', labelsize=35)

    # Create scatter plot for MS stars
    ax.plot(B1_MS, A1_MS, marker='o', linestyle='-', markersize=10, color='blue', alpha=0.7, label='Main Sequence')

    # Create scatter plot for giants
    ax.plot(B1_giant, A1_giant, marker='o', linestyle='-', markersize=10, color='red', alpha=0.7, label='Giants')

    # Legend
    ax.legend(fontsize=20)

plt.savefig("color-color-diagram.pdf")
