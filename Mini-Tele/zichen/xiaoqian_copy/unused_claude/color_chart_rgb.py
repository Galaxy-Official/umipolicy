import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def create_blue_to_magenta_chart():
    """Create RGB color chart from blue to magenta with 11 colors"""

    # Define start (blue) and end (magenta) colors in RGB
    blue = np.array([0, 0, 255])  # Pure blue
    magenta = np.array([255, 0, 255])  # Magenta (purple-red)

    # Create 11 colors by interpolating between blue and magenta
    colors = []
    for i in range(11):
        # Linear interpolation factor (0 to 1)
        t = i / 10.0
        # Interpolate RGB values
        color = blue + t * (magenta - blue)
        colors.append(color / 255.0)  # Normalize to 0-1 range for matplotlib

    return colors

def display_color_chart(colors):
    """Display the color chart with RGB values"""

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8),
                                   gridspec_kw={'height_ratios': [1, 2]})

    # Top panel: color swatches
    for i, color in enumerate(colors):
        ax1.add_patch(patches.Rectangle((i, 0), 1, 1,
                                       facecolor=color, edgecolor='black'))

    ax1.set_xlim(0, 11)
    ax1.set_ylim(0, 1)
    ax1.set_xticks(range(11))
    ax1.set_xticklabels([f'Color {i+1}' for i in range(11)], rotation=45)
    ax1.set_yticks([])
    ax1.set_title('Blue to Magenta Color Gradient (11 Colors)', fontsize=14, fontweight='bold')

    # Bottom panel: RGB values table
    ax2.axis('tight')
    ax2.axis('off')

    # Create table data
    table_data = []
    headers = ['Color #', 'R', 'G', 'B', 'Hex']

    for i, color in enumerate(colors):
        rgb_values = [int(c * 255) for c in color]
        hex_color = '#{:02x}{:02x}{:02x}'.format(*rgb_values)
        table_data.append([f'Color {i+1}', rgb_values[0], rgb_values[1], rgb_values[2], hex_color])

    table = ax2.table(cellText=table_data, colLabels=headers,
                     cellLoc='center', loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.5)

    # Color the hex code cells with their actual colors
    for i in range(11):
        rgb_values = [int(c * 255) for c in colors[i]]
        hex_color = '#{:02x}{:02x}{:02x}'.format(*rgb_values)
        table[(i+1, 4)].set_facecolor(colors[i])
        # Set text color to white for darker colors
        if sum(rgb_values) < 384:  # Threshold for dark colors
            table[(i+1, 4)].set_text_props(color='white')

    plt.tight_layout()
    plt.savefig('blue_to_magenta_color_chart.png', dpi=300, bbox_inches='tight')
    plt.show()

    # Print RGB values to console
    print("Blue to Magenta RGB Color Chart (11 colors)")
    print("=" * 50)
    for i, color in enumerate(colors):
        rgb_values = [int(c * 255) for c in color]
        hex_color = '#{:02x}{:02x}{:02x}'.format(*rgb_values)
        print(f"Color {i+1:2d}: RGB({rgb_values[0]:3d}, {rgb_values[1]:3d}, {rgb_values[2]:3d}) - {hex_color}")

def save_color_data(colors):
    """Save color data to CSV file"""
    import csv

    with open('blue_to_magenta_colors.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Color_Index', 'R', 'G', 'B', 'Hex'])

        for i, color in enumerate(colors):
            rgb_values = [int(c * 255) for c in color]
            hex_color = '#{:02x}{:02x}{:02x}'.format(*rgb_values)
            writer.writerow([i+1, rgb_values[0], rgb_values[1], rgb_values[2], hex_color])

    print("\nColor data saved to 'blue_to_magenta_colors.csv'")

if __name__ == "__main__":
    # Generate the color chart
    colors = create_blue_to_magenta_chart()

    # Display and save
    display_color_chart(colors)
    save_color_data(colors)