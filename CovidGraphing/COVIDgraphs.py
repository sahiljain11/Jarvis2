

#import initExample ## Add path to library (just for examples; you do not need this)


from pyqtgraph.Qt import QtGui, QtCore
import numpy as np
import pyqtgraph as pg
import pandas as pd

#read data
URL_DATASET = r'https://raw.githubusercontent.com/datasets/covid-19/master/data/countries-aggregated.csv'
df1 = pd.read_csv(URL_DATASET, parse_dates=['Date'])
first_date = df1.loc[0, 'Date']
df1['Date'] = (df1['Date'] - first_date).dt.days

#obtain data for countries

df_canada = df1[df1['Country'] == 'Canada']
df_canada = df_canada.tail(20)
df_us = df1[df1['Country'] == 'US']
df_us = df_us.tail(20)
df_mexico = df1[df1['Country'] == 'Mexico']
df_mexico = df_mexico.tail(20)
df_guat = df1[df1['Country'] == 'Guatemala']
df_guat = df_guat.tail(20)
df_cuba = df1[df1['Country'] == 'Cuba']
df_cuba = df_cuba.tail(20)
df_haiti = df1[df1['Country'] == 'Haiti']
df_haiti = df_haiti.tail(20)
df_domrep = df1[df1['Country'] == 'Dominican Republic']
df_domrep = df_domrep.tail(20)


#QtGui.QApplication.setGraphicsSystem('raster')
app = QtGui.QApplication([])
#mw = QtGui.QMainWindow()
#mw.resize(800,800)

win = pg.GraphicsWindow(title="North America and the World, COVID-19")
win.resize(1000,700)


x1 = df_canada['Confirmed'].tolist()
x2 = df_us['Confirmed'].tolist()
x3 = df_mexico['Confirmed'].tolist()
x4 = df_guat['Confirmed'].tolist()
x5 = df_cuba['Confirmed'].tolist()
x6 = df_haiti['Confirmed'].tolist()
x7 = df_domrep['Confirmed'].tolist()


y1 = df_canada['Date'].tolist()
y2 = df_us['Date'].tolist()
y3 = df_mexico['Date'].tolist()
y4 = df_guat['Date'].tolist()
y5 = df_cuba['Date'].tolist()
y6 = df_haiti['Date'].tolist()
y7 = df_domrep['Date'].tolist()


p2 = win.addPlot(title="North America Confirmed Cases")
p2.setWindowTitle('Legend') #you can drag and move the legend, and zoom the graph
p2.addLegend()
p2.setLabel('left', "Number of Confirmed Cases")
p2.setLabel('bottom', "Days since January 22, 2020")
p2.plot(y1, x1, pen=(255,0,0), name="Canada") # Just for testing
p2.plot(y2, x2, pen=(255,127,0), name="USA")
p2.plot(y3, x3, pen=(255, 255,0), name="Mexico")
p2.plot(y4, x4, pen=(0,255,0), name="Guatemala")
p2.plot(y5, x5, pen=(0,255,255), name="Cuba")
p2.plot(y6, x6, pen=(255,235,205), name="Haiti")
p2.plot(y7, x7, pen=(143,0,255), name="Dominican Republic")





win.nextRow()

#next graph

#extract data
df_us = df1[df1['Country'] == 'US']
df_us = df_us.tail(200)
df_brazil = df1[df1['Country'] == 'Brazil']
df_brazil = df_brazil.tail(200)
df_russia = df1[df1['Country'] == 'Russia']
df_russia = df_russia.tail(200)
df_uk = df1[df1['Country'] == 'United Kingdom']
df_uk = df_uk.tail(200)
df_china = df1[df1['Country'] == 'China']
df_china = df_china.tail(200)

#load to variables as list
x8 = df_us['Confirmed'].tolist()
x9 = df_brazil['Confirmed'].tolist()
x10 = df_russia['Confirmed'].tolist()
x11 = df_uk['Confirmed'].tolist()
x12 = df_china['Confirmed'].tolist()

y8 = df_us['Date'].tolist()
y9 = df_brazil['Date'].tolist()
y10 = df_russia['Date'].tolist()
y11 = df_uk['Date'].tolist()
y12 = df_china['Date'].tolist()

#create plot
p6 = win.addPlot(title="USA and the World, Confirmed Cases")
p6.resize(1000,700)
p6.setWindowTitle('Legend') #you can drag and move the legend, and zoom the graph
p6.addLegend()
p6.setLabel('left', "Number of Confirmed Cases")
p6.setLabel('bottom', "Days since January 22, 2020")
p6.plot(y8, x8, pen=(255,0,0), name="USA") # Just for testing
p6.plot(y9, x9, pen=(255,127,0), name="Brazil")
p6.plot(y10, x10, pen=(255, 255,0), name="Russia")
p6.plot(y11, x11, pen=(0,255,0), name="United Kingdom")
p6.plot(y12, x12, pen=(0,255,255), name="China")








## Start Qt event loop unless running in interactive mode or using pyside.
if __name__ == '__main__':
    import sys
    if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
        QtGui.QApplication.instance().exec_()
