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
from sklearn.metrics import confusion_matrix,accuracy_score,classification_report

def plot_IAHOS(y,ogp,ogp2,tgp,tgp2):
    fig = make_subplots(rows=2, cols=2,subplot_titles=("Mean train. accuracy first round",
                                                   "Mean valid accuracy first round",
                                                  "Mean train accuracy last round",
                                                  "Mean valid accuracy last round"))

    x = np.linspace(0,len(tgp[0])-1,len(tgp[0]))
    Colorscale = [[0, '#FF0000'],[0.5, '#F1C40F'], [1, '#00FF00']]
    fig.add_trace(go.Heatmap(y=[y[i] for i in range(len(y))],
                       x=[0,1],
                       z=ogp2, colorscale = Colorscale),row=1,col=1)
    fig.add_trace(go.Heatmap(y=[y[i] for i in range(len(y))],
                       x=[0,1],
                       z=ogp,colorscale=Colorscale),row=1,col=2)
    fig.add_trace(go.Heatmap(y=[y[i] for i in range(len(y))],
                       x=x,
                       z=tgp2, colorscale = Colorscale),row=2,col=1)
    fig.add_trace(go.Heatmap(y=[y[i] for i in range(len(y))],
                       x=x,
                       z=tgp,colorscale=Colorscale),row=2,col=2)
    fig.update_layout(height=600, width=800)
    if not os.path.exists("images"):
        os.mkdir("images")
    fig.write_image('images/IAHOS.png')
    fig.show()

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
    
def plot_earth_clustering(training_set,clusters_index,d,e,what):    
    fig = plt.figure(figsize=(14, 8))
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.Robinson())
    
    ax.set_global()
    
    ax.stock_img()
    ax.coastlines()
    points = np.ndarray(shape=(len(training_set),2),dtype=float) #this vector contains latitude and longitude of each location added 
    #in the 'e' dictionary
    i = 0
    for j in training_set: #for every locations that I put in the 'e' dictionary:
        points[i,0] = d[j].iat[0,6]
        points[i,1] = d[j].iat[0,7]
        i=i+1
    colors=['red','yellow','green','purple']
    ax.scatter([points[i,1] for i in range(len(training_set))],
               [points[i,0] for i in range(len(training_set))],
               c=[colors[clusters_index[i]] for i in range(len(training_set))],
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
    
def plot_confusion_matrix(new_test_labels,y_pred,words_name,model):
    cm = confusion_matrix(y_true=new_test_labels,y_pred=y_pred)
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    plt.figure(figsize=(6,6))
    plt.imshow(cm,interpolation='nearest',cmap=plt.cm.jet)
    plt.tight_layout()
    plt.colorbar()
    tick_marks = np.arange(len(words_name))
    plt.xticks(tick_marks,words_name,rotation=90)
    plt.yticks(tick_marks,words_name)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion matrix')
    if not os.path.exists("images"):
        os.mkdir("images")
    plt.savefig('images/confusion_matrix_'+str(model)+'.png')
    plt.show()
    print(classification_report(new_test_labels, y_pred, target_names=words_name))
    
def graph(key,d,e):    
    values = e[key]#import all the rows corresponding to that location
    plotly.offline.init_notebook_mode()
    x1 = np.arange(1,365)
    Tmin = np.array(values.T)[0]
    Tmax = np.array(values.T)[1]
    Rain = np.array(values.T)[3]/100
    Snow = np.array(values.T)[4]
    Snwd = np.array(values.T)[5]/10
    Tavg = np.array(values.T)[6]
    trace0 = go.Scatter(x=x1,y=Tmin,name="Min Temperature",mode="lines+markers")
    trace1 = go.Scatter(x=x1,y=Tmax,name="Max Temperature",mode="lines+markers")
    trace2 = go.Scatter(x=x1,y=Tavg,name="Average temperature",mode="lines+markers")
    trace3 = go.Bar(x=x1,y=Rain,name="Daily rain precipitations (mm)")
    K1=1
    K2=1
    for i in range(365):
        if Snow[i]>50:
            K1=10
    for i in range(365):
        if Snwd[i]>50:
            K2=10
    if K1 == 1:
        trace4 = go.Bar(x=x1,y=Snow/K1,name="Daily snow precipitations")
    else:
        trace4 = go.Bar(x=x1,y=Snow/K1,name="Daily snow precipitations [tenths of mm]")
    if K2 == 1:
        trace5 = go.Bar(x=x1,y=Snwd/K2,name="Daily snowfall")
    else:
        trace5 = go.Bar(x=x1,y=Snwd/K2,name="Daily snowfall [tenths of mm]")
    mydata = go.Data([trace0, trace1, trace2, trace3,trace4,trace5])
    mylayout = go.Layout(title="Weather conditions in "+d[key].iloc[0]['Location'],xaxis=dict(title='Days of the year'),
                        height=400,width=800)
    
    fig = go.Figure(data=mydata, layout=mylayout)
    if not os.path.exists("images"):
        os.mkdir("images")
    fig.write_image('images/Weather '+str(key)+'.png')
    plotly.offline.iplot(fig, filename = d[key].iloc[0]['Location'])