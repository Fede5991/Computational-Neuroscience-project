def construction(n,q1,q2,q3,q4,q5,q6):
#n corresponds to the number of locations added to the dictionary. q1,...q6 are the values that I can use to filter
#only those locations that have enough informations during the year. If q1=1 means that I want to obatin those places
#that have completely infomation about minimum temperature, e.g. a value for each day of the year (q1=0 I have 0 values
#of that feature during the year); q2 for maximum temperature, q3 for rain precipitation, q4 for snow fall, q5 for snow depth
#and q6 for average daily temperature
    x = {}#create dictionray containing weather informations of each location inserted in a proper matrix
    y = {}#create dictionary containing informations about the quantity of values that I have for each location
    for i in range(n):
            location2 = np.ndarray(shape=(365,7), dtype=float)#create the matrix
            key = Locations[samples[i]]#extract the Code of the location
            location = d[key].values#extract all informations available corresponding to that Code
            quality_features=np.zeros(6)#create the vectors used to represent the quantity of informations of each place
            Time_extended = location[:,1]
            Time = np.asarray(Time_extended) #transform to array
            Time = np.unique(Time) #this array contains the dates without repetitions
            Tmin = np.zeros(Time.size) #array containing min daily temperatures (C° degrees)
            Tmax = np.zeros(Time.size) #array containing max daily temperatures (C° degrees)
            Rain = np.zeros(Time.size) #array containing daily rain precipitations (mm)
            Snow = np.zeros(Time.size) #array containing daily snow precipitations (mm)
            Snwd = np.zeros(Time.size) #array containing daily snow precipitations (mm)
            Tavg = np.zeros(Time.size) #array containing daily snow precipitations (mm)

            i1=0#counter for minimum temperature
            i2=0#counter for maximum temperature
            i3=0#counter for rain 
            i4=0#counter for snow fall
            i5=0#counter for snow depth
            i6=0#counter for average temperature

            i7=0#counter for the number of samples of minimum temperature
            i8=0#counter for the number of samples of maximum temperature
            i9=0#counter for the number of sampels of rain 
            i10=0#counter for the number of samples of snowfall
            i11=0#counter for the number of samples of snow depth
            i12=0#counter for the number of samples of average temperature


            for i in range(Time_extended.size-1):
                if (location[i,2] == "TMIN"):
                    i7=i7+1#increment the iterator corresponding to the number of sample of Tmin
                    Tmin[i1] = location[i,3]/10#because in the dataset temperature is divided by 10           
                    if (location[i,1] % 100) != (location[i+1,1] % 100):#if the date of this row is different
                        i1=i1+1#from the date of the previous row, I increment all the iterators 
                        i2=i2+1
                        i3=i3+1
                        i4=i4+1
                        i5=i5+1
                        i6=i6+1
                else:
                    if (location[i,2] == "TMAX"):
                        i8=i8+1#increment the iterator corresponding to number of samples of Tmax
                        Tmax[i2] = location[i,3]/10#because in the dataset temperature is divided by 10
                        if (location[i,1] % 100) != (location[i+1,1] % 100):#if the date of this row is 
                            i1=i1+1#different from the previous one, I increment all the iterators
                            i2=i2+1
                            i3=i3+1
                            i4=i4+1
                            i5=i5+1
                            i6=i6+1
                    else:
                        if (location[i,2] == "PRCP"):
                            i9=i9+1
                            Rain[i3] = location[i,3]
                            if (location[i,1] % 100) != (location[i+1,1] % 100):
                                i1=i1+1
                                i2=i2+1
                                i3=i3+1
                                i4=i4+1
                                i5=i5+1
                                i6=i6+1
                        else:
                            if (location[i,2] == "SNOW"):
                                i10=i10+1
                                Snow[i4] = location[i,3]
                                if (location[i,1] % 100) != (location[i+1,1] % 100):
                                    i1=i1+1
                                    i2=i2+1
                                    i3=i3+1
                                    i4=i4+1
                                    i5=i5+1
                                    i6=i6+1
                            else:
                                if (location[i,2] == "SNWD"):
                                    i11=i11+1
                                    Snwd[i5] = location[i,3]    
                                    if (location[i,1] % 100) != (location[i+1,1] % 100):                            
                                        i1=i1+1
                                        i2=i2+1
                                        i3=i3+1
                                        i4=i4+1
                                        i5=i5+1
                                        i6=i6+1
                                else:
                                    if (location[i,2] == "TAVG"):
                                        i12=i12+1
                                        Tavg[i6] = location[i,3]/10    
                                        if (location[i,1] % 100) != (location[i+1,1] % 100):                            
                                            i1=i1+1
                                            i2=i2+1
                                            i3=i3+1
                                            i4=i4+1
                                            i5=i5+1
                                            i6=i6+1
                                    else:
                                        if (location[i,1] % 100) != (location[i+1,1] % 100):                            
                                            i1=i1+1
                                            i2=i2+1
                                            i3=i3+1
                                            i4=i4+1
                                            i5=i5+1
                                            i6=i6+1
            quality_features[0]=i7/365#normalize between 0 and 1 the number of samples that I have for each feature
            quality_features[1]=i8/365
            quality_features[2]=i9/365
            quality_features[3]=i10/365 
            quality_features[4]=i11/365
            quality_features[5]=i12/365 

            z = 0
            for i in range(Time.size):#put all the features in an unique matrix, location2
                    location2[z,0]=Tmin[i]
                    location2[z,1]=Tmax[i]
                    location2[z,2]=i
                    location2[z,3]=Rain[i]
                    location2[z,4]=Snow[i]
                    location2[z,5]=Snwd[i]
                    location2[z,6]=Tavg[i]
                    z = z+1
#if the locations have enough samples w.r.t. the limits tha I impose with the numbers q1,q2,q3,q4,q5 and q6,            
#I add the matrix location2 to the x-dictionary and the quality-features vector to the y-dictionary          
            if (quality_features[0]>=q1) and (quality_features[1]>=q2) and (quality_features[2]>=q3) and (quality_features[3]>=q4) and (quality_features[4]>=q5) and (quality_features[5]>=q6):
                x[key]=location2
                y[key]=quality_features
    return x,y            