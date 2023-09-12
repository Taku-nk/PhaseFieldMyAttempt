
if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import numpy as np
    import sys
    sys.path.append("C:\\Users\\taku\\Hiroshima-U-Master\\OneDrive - Hiroshima University\\ドキュメント\\1kouza\\MasterResearch\\PythonCode\\ReadExcel2D")
    from VizExcel import ExcelViewer, PlotExcel

    sys.path.append("C:\\Users\\taku\\Hiroshima-U-Master\\OneDrive - Hiroshima University\\ドキュメント\\1kouza\\MasterResearch\\PythonCode\\visualize_2D")
    from visualize_DEM_2D import debug_visualize_mesh as viz

    import pandas as pd

    iter_id = 1
    file_path = f'./test_single_notch/output/iter_{iter_id}/DEM_original_result.csv'

    df = pd.read_csv(file_path)

    x = df.loc[:, 'x'].to_numpy()
    y = df.loc[:, 'y'].to_numpy()


    # val = df.loc[:, 'phi'].to_numpy()
    # val = df.loc[:, 'elastic_energy_density'].to_numpy()
    # val = df.loc[:, 'fracture_energy_density'].to_numpy()
    # val = df.loc[:, 'disp_x'].to_numpy()
    val = df.loc[:, 'disp_y'].to_numpy()
    # val = df.loc[:, 'disp_x'].to_numpy()

    # val = df.loc[:, 'stress_x'].to_numpy()



    viz(x, y, val, 50)
    # viz(x, y, val, np.linspace(0.99, 1.01, 2))

