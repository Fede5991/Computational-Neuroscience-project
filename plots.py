# -*- coding: utf-8 -*-
"""
Created on Sun Sep 15 17:53:58 2019

@author: Fede
"""
import matplotlib.pyplot as plt
import cartopy #to draw the world map
import cartopy.crs as ccrs
import plotly #to create graphs
import plotly.graph_objs as go
from plotly.subplots import make_subplots
import numpy as np
import os

def plot_earth_trainvaltest(training_set,validation_set,test_set,d):
    fig = plt.figure(figsize=(14, 8))
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.Robinson())
    
    ax.set_global()
    
    ax.stock_img()
    ax.coastlines()
    points = np.ndarray(shape=(len(training_set),2),dtype=float) #this vector contains latitude and longitude of each location added 
    points2 = np.ndarray(shape=(len(validation_set),2),dtype=float)
    points3 = np.ndarray(shape=(len(test_set),2),dtype=float)
    #in the 'e' dictionary
    i = 0
    for j in training_set: #for every locations that I put in the 'e' dictionary:
        points[i,0] = d[j].iat[0,6]
        points[i,1] = d[j].iat[0,7]
        i=i+1
    
    i = 0
    for j in validation_set: #for every locations that I put in the 'e' dictionary:
        points2[i,0] = d[j].iat[0,6]
        points2[i,1] = d[j].iat[0,7]
        i=i+1    
    i = 0
    for j in test_set: #for every locations that I put in the 'e' dictionary:
        points3[i,0] = d[j].iat[0,6]
        points3[i,1] = d[j].iat[0,7]
        i=i+1
    
    ax.scatter([points[i,1] for i in range(len(training_set))],
               [points[i,0] for i in range(len(training_set))],
               color='red',
               transform=ccrs.Geodetic())
    ax.scatter([points2[i,1] for i in range(len(validation_set))],
               [points2[i,0] for i in range(len(validation_set))],
               color='yellow',
               transform=ccrs.Geodetic())
    ax.scatter([points3[i,1] for i in range(len(test_set))],
               [points3[i,0] for i in range(len(test_set))],
               color='green',
               transform=ccrs.Geodetic())
    if not os.path.exists("images"):
        os.mkdir("images")
    plt.savefig('images/trainvaltestearth')
    plt.show()

def plot_earth(d,e,what):    
    fig = plt.figure(figsize=(14, 8))
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.Robinson())
    
    ax.set_global()
    
    ax.stock_img()
    ax.coastlines()
    points = np.ndarray(shape=(len(e),2),dtype=float) #this vector contains latitude and longitude of each location added 
    #in the 'e' dictionary
    i = 0
    for j in e: #for every locations that I put in the 'e' dictionary:
        points[i,0] = d[j].iat[0,6]
        points[i,1] = d[j].iat[0,7]
        i=i+1
    colors=['red','yellow']
    ax.scatter([points[i,1] for i in range(len(e))],
               [points[i,0] for i in range(len(e))],
               color='red',
               transform=ccrs.Geodetic())
    if not os.path.exists("images"):
        os.mkdir("images")
    plt.savefig('images/earth'+str(what))
    plt.show()

def plot_split(labels,values1,values2,values3):    
    fig = make_subplots(rows=1, cols=3,specs=[[{"type": "domain"}, {"type": "domain"},
               {"type": "domain"}]])
    fig.add_trace(go.Pie(labels=labels, values=values1),row=1,col=1)
    fig.add_trace(go.Pie(labels=labels, values=values2),row=1,col=2)
    fig.add_trace(go.Pie(labels=labels, values=values3),row=1,col=3)
    fig.update_layout(height=300,width=700,title_text='Training/Val/Test distributions of the countries')
    if not os.path.exists("images"):
        os.mkdir("images")
    fig.write_image('images/dataset_split.png')
    fig.show()

def plot_quality(d,f,extra_input):    
    w = []
    names = np.empty(len(f), dtype='object')
    r = 0
    for i in f:
        w.append(list(f[i]))
        names[r] = d[i].iloc[0].Location
        #y[r] = i
        r=r+1
        #y = [",".join(item) for item in y.astype(str)]
    #date_list = [dates[x] for x in range(0, len(f))]
    plotly.offline.init_notebook_mode()
    
    Colorscale = [[0, '#FF0000'],[0.5, '#F1C40F'], [1, '#00FF00']]
    trace = go.Heatmap(y=[names[i] for i in range(0,len(f))],
                       x=['Tmin', 'Tmax', 'Rain', 'Snowfall', 'Snowdepth','Tavg'],
                       z=[w[i] for i in range(len(f))], colorscale = Colorscale)
    mydata=[trace]
    layout=go.Layout(autosize=False, height=500, width=700)
    fig = go.Figure(data=mydata,layout=layout)
    if not os.path.exists("images"):
        os.mkdir("images")
    fig.write_image('images/quality '+str(extra_input)+'.png')
    plotly.offline.iplot(fig, filename = 'Quality features of the dataset')
    #the columns represents the input features and the rows the locations. 