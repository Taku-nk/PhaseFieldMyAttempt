"""Debug Snippet for inspecting the code.

Just copy and paste the following snippet in the debug consle during debugging.


--------------------------------------------------------------------------------
For 2D visualization
--------------------------------------------------------------------------------
import matplotlib.pyplot as plt
fig=plt.figure()
ax = fig.add_subplot()
ax.axis('equal')
ax.grid(ls=':')

ax.scatter()
ax.clear()


--------------------------------------------------------------------------------
For 3D visualization
--------------------------------------------------------------------------------
import matplotlib.pyplot as plt
fig=plt.figure()
ax = fig.add_subplot(projection='3d')

ax.scatter()
ax.clear()


# vor_polys visualizer
for cell in vor_polys:
    vertCoord = np.array(cell)
    ax.fill(vertCoord[:, 0], vertCoord[:, 1])

    
--------------------------------------------------------------------------------
debug_visualize_mesh
--------------------------------------------------------------------------------
import sys
sys.path.append("C:\\Users\\taku\\Hiroshima-U-Master\\OneDrive - Hiroshima University\\ドキュメント\\1kouza\\MasterResearch\\PythonCode\\visualize_2D")
from visualize_DEM_2D import debug_visualize_mesh as viz

viz()
"""