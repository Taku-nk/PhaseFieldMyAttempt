
if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import numpy as np
    import sys
    sys.path.append("C:\\Users\\taku\\Hiroshima-U-Master\\OneDrive - Hiroshima University\\ドキュメント\\1kouza\\MasterResearch\\PythonCode\\ReadExcel2D")
    from VizExcel import ExcelViewer, PlotExcel

    sys.path.append("C:\\Users\\taku\\Hiroshima-U-Master\\OneDrive - Hiroshima University\\ドキュメント\\1kouza\\MasterResearch\\PythonCode\\visualize_2D")
    from visualize_DEM_2D import debug_visualize_mesh as viz

    import pandas as pd

    # file_path = 'test_1results.xlsx'
    # file_path = 'test_01_vary_on_fill_on_with_overlay_func\DEM_original_result.csv'
    file_path = './test_small_hole/DEM_original_result.csv'
    # key = 'S12'
    df = pd.read_csv(file_path)

    x = df.loc[:, 'x'].to_numpy()
    y = df.loc[:, 'y'].to_numpy()


    val = df.loc[:, 'stress_x'].to_numpy()
    # val = df.loc[:, 'disp_x'].to_numpy()

    # b =plt.scatter(x, y, c=val)
    # plt.colorbar(b)
    # plt.show()
    # key = 'v'


    viz(x, y, val, 50)
    # viz(x, y, val, np.linspace(0.99, 1.01, 2))

    # excel_result = ExcelViewer(file_path)
    # # print(excel_result.GetDataFRame())

    # plotter = PlotExcel()
    # plotter.plotResult(excel_result, key=key)
    # plotter.plotResult(excel_result, key='S11', hole_radius=None)
