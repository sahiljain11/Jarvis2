

#import initExample ## Add path to library (just for examples; you do not need this)


from pyqtgraph.Qt import QtGui, QtCore
import numpy as np
import pyqtgraph as pg
import pandas as pd
URL_DATASET = r'https://raw.githubusercontent.com/datasets/covid-19/master/data/countries-aggregated.csv'
df1 = pd.read_csv(URL_DATASET)
df_canada = df1[df1['Country'] == 'Canada']

df_us = df1[df1['Country'] == 'US']

df_mexico = df1[df1['Country'] == 'Mexico']
list_dates = df1['Date'].unique()



#QtGui.QApplication.setGraphicsSystem('raster')
app = QtGui.QApplication([])
#mw = QtGui.QMainWindow()
#mw.resize(800,800)

win = pg.GraphicsWindow(title="North America and the World, COVID-19")
win.resize(1000,600)


x1 = df_canada['Confirmed']
# x2 = df_us['Confirmed']
# x3 = df_mexico['Confirmed']
df_canada['Date'] = df_canada['Date'].map(lambda x: x.lstrip('-'))

# y2 = df_us['Date']
# y3 = df_mexico['Date']

p2 = win.addPlot(title="North America Confirmed Cases")
p2.plot(x1, df_canada['Date']) # Just for testing
# p2.plot(x2, y2)
# p2.plot(x3, y3)



# p2 = win.addPlot(title="North America Confirmed Cases")
# p2.plot(x1, y1)
# p2.plot(x2, y2)
# p2.plot(x3, y3)



# p6 = win.addPlot(title="Live World Confirmed Cases")
# curve = p6.plot(pen='y')
# data = np.random.normal(size=(10,1000))
# ptr = 0
# def update():
#     global curve, data, ptr, p6
#     curve.setData(data[ptr%10])
#     if ptr == 0:
#         p6.enableAutoRange('xy', False)  ## stop auto-scaling after the first data set is plotted
#     ptr += 1
# timer = QtCore.QTimer()
# timer.timeout.connect(update)
# timer.start(50)


# win.nextRow()




## Start Qt event loop unless running in interactive mode or using pyside.
if __name__ == '__main__':
    import sys
    if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
        QtGui.QApplication.instance().exec_()