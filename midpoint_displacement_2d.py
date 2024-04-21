import random
from collections import deque

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

# All the credit goes to the author of *64bitdragon* blog:
# https://learn.64bitdragon.com/articles/computer-science/procedural-generation/midpoint-displacement-in-two-dimensions
# I just changed the code to use numpy, used more descriptive variable names, and added a visualization.


def midpoint_displacement_2d(n: int = 8, roughness: int = 200) -> np.ndarray:
    """Midpoint displacement algorithm in 2D. Generates a heightmap using the midpoint displacement algorithm.
    
    Parameters
    ----------
    n : int
        The number of iterations. The size of the grid is $2^n + 1$.
    roughness : int
        The roughness of the terrain. The higher the value, the rougher the terrain.
    
    Returns
    -------
    np.ndarray
        A 2D numpy array representing the heightmap.
    """
    size = 2**n + 1
    heightmap = np.zeros((size, size))

    # Initialize corners.
    heightmap[0, 0] = random.randint(0, 256)
    heightmap[0, size - 1] = random.randint(0, 256)
    heightmap[size - 1, 0] = random.randint(0, 256)
    heightmap[size - 1, size - 1] = random.randint(0, 256)

    q = deque()

    # Add the initial square to the queue (x0, y0, x1, y1, randomness).
    q.append((0, 0, size - 1, size - 1, roughness))

    def my_randint(roughness: int) -> int:
        return random.randint(-roughness, roughness)

    while len(q) != 0:
        # (x0, y0), (x1, y1) cartesian coordinates of the square (top-left, bottom-right).
        x0, y0, x1, y1, roughness = q.popleft()

        # Calculate the center point of the square.
        cy = (y0 + y1) // 2
        cx = (x0 + x1) // 2

        # Get the corner values.
        top_left = heightmap[y0, x0]
        top_right = heightmap[y0, x1]
        bottom_left = heightmap[y1, x0]
        bottom_right = heightmap[y1, x1]

        # Left midpoint
        heightmap[cy, x0] = (top_left + bottom_left) // 2 + my_randint(roughness)
        # Right midpoint
        heightmap[cy, x1] = (top_right + bottom_right) // 2 + my_randint(roughness)
        # Top midpoint
        heightmap[y0, cx] = (top_left + top_right) // 2 + my_randint(roughness)
        # Bottom midpoint
        heightmap[y1, cx] = (bottom_left + bottom_right) // 2 + my_randint(roughness)
        # Center midpoint
        heightmap[cy, cx] = (
            top_left + bottom_left + top_right + bottom_right
        ) // 4 + my_randint(roughness)

        # If the width of the segment is greater than 2, then it can be subdivided.
        if y1 - y0 > 2:
            roughness //= 2  # Reduce the roughness.
            # Add the 4 squares to the queue.
            #
            # (x0,y0)
            #        ┌----┬----┐
            #        | 1  | 2  |
            #        ├----┼----┤ cy
            #        | 3  | 4  |
            #        └----┴----┘
            #            cx     (x1,y1)
            q.append((x0, y0, cx, cy, roughness))  # 1
            q.append((cx, y0, x1, cy, roughness))  # 2
            q.append((x0, cy, cx, y1, roughness))  # 3
            q.append((cx, cy, x1, y1, roughness))  # 4

    return heightmap


def plot_terrain(heightmap: np.ndarray, sea_level: int = 25) -> None:
    """Plot the 2D terrain heightmap.

    Parameters
    ----------
    heightmap : np.ndarray
        A 2D numpy array representing the heightmap.
    sea_level : int
        The sea level. All the terrain below this level will be clipped.
    """

    # Clip the terrain to the see level
    heightmap[heightmap < sea_level] = sea_level

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection="3d")
    x = np.arange(0, heightmap.shape[0])
    y = np.arange(0, heightmap.shape[1])
    X, Y = np.meshgrid(x, y)
    surf = ax.plot_surface(X, Y, heightmap, cmap="gist_earth")  # or 'terrain'
    fig.colorbar(surf, shrink=0.5)
    ax.set_title("Midpoint Displacement 2D")
    ax.set_xlabel("$x$")
    ax.set_ylabel("$y$")
    ax.set_zlabel("$z$")
    ax.view_init(elev=20, azim=240)
    plt.show()


if __name__ == "__main__":
    random.seed(42)
    n = 9
    roughness = 200
    heightmap = midpoint_displacement_2d(n, roughness)
    plot_terrain(heightmap, sea_level=50)
