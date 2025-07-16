import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

def visualize_npy(npy_file_path):
    """
    Visualizes 3D volumetric data from a .npy file.

    Parameters:
    npy_file_path (str): Path to the .npy file to be visualized.
    """
    # Load the .npy file
    volume = np.load(npy_file_path)

    # Create figure and subplot
    fig, ax = plt.subplots()
    plt.subplots_adjust(bottom=0.25)

    # Initialize display with the first slice
    slice_index = 0
    img = ax.imshow(volume[slice_index, :, :], cmap='gray')

    # Create a slider
    ax_slider = plt.axes([0.2, 0.1, 0.65, 0.03])
    slider = Slider(ax_slider, 'Slice', 0, volume.shape[0] - 1, valinit=0, valstep=1)

    # Define update function for the slider
    def update(val):
        slice_index = int(slider.val)
        img.set_data(volume[slice_index, :, :])
        fig.canvas.draw_idle()

    # Connect the update function to the slider
    slider.on_changed(update)

    plt.show()

if __name__ == "__main__":
    # Replace with your .npy file path
    npy_file_path = "./npy/volume.npy"
    visualize_npy(npy_file_path)
