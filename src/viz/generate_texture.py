import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

def generate_texture(width=1024, height=512, filename='earth_texture.jpg'):
    print("Generating procedural texture...")
    # Generate random noise
    
    # Perlin-ish noise would be better, but let's do smoothed random noise
    # Simple approach: Blur a random matrix
    from scipy.ndimage import gaussian_filter
    
    # Base noise
    noise = np.random.rand(height, width)
    
    # Smooth it to create "continents"
    # Sigma controls the size of blobs
    sigma = 15
    smoothed = gaussian_filter(noise, sigma=sigma)
    
    # Normalize
    smoothed = (smoothed - smoothed.min()) / (smoothed.max() - smoothed.min())
    
    # Create colors
    # < 0.5: Ocean (Deep blue to light blue)
    # > 0.5: Land (Green to Brown)
    
    # Custom Colormap
    colors = [(0, 'darkblue'), (0.5, 'deepskyblue'), (0.55, 'forestgreen'), (0.7, 'saddlebrown'), (1.0, 'white')]
    cm = LinearSegmentedColormap.from_list("earth_map", colors)
    
    # Apply colormap
    texture_rgba = cm(smoothed)
    
    # Save using matplotlib
    plt.imsave(filename, texture_rgba)
    print(f"Texture saved to {filename}")

if __name__ == "__main__":
    try:
        generate_texture()
    except Exception as e:
        print(f"Failed to generate texture: {e}")
