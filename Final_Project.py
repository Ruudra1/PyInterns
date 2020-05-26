import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sqlite3
import seaborn as sns
from sklearn.linear_model import LinearRegression
import tkinter as tk
from tkinter import *
from tkinter.ttk import *
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib
import random

'''
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error # 0.3 error
from sklearn.model_selection import train_test_split
#matplotlib.use("TkAgg")
'''
#from matplotlib.figure import Figure
#from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

con = sqlite3.connect('C:\\Users\\Ruudra\\Desktop\\Python Internship\\DB1.db',isolation_level = None)
curr = con.cursor()
#curr.execute('SELECT Category,sum(Installs) FROM CleanedData GROUP by Category;')

#setting the console width 
pd.set_option('display.max_columns',20)
pd.set_option('display.max_rows',200)
pd.set_option('display.max_colwidth',100)
pd.set_option('display.width',1000)

'''
df = pd.DataFrame()
df = pd.read_csv('C:\\Users\\Ruudra\\Desktop\\Python Internship\\CleanedData.csv')

#Data Cleaning For Installs column
df['Installs'] = df['Installs'].map(lambda x: x.rstrip('+'))
df['Installs'] = df['Installs'].map(lambda x: ''.join(x.split(',')))
df['Installs'] = df['Installs'].map(lambda x: x.rstrip('+'))

# Data cleaning for "Price" column
df['Price'] = df['Price'].map(lambda x: x.lstrip('$').rstrip())

# Data cleaning for "Size" column
df['Size'] = df['Size'].map(lambda x: x.rstrip('M'))
df['Size'] = df['Size'].map(lambda x: str(round((float(x.rstrip('k'))/1024), 1)) if x[-1]=='k' else x)
df['Size'] = df['Size'].map(lambda x: np.nan if x.startswith('Varies') else x)

#print(df.info())

# Sort by "Category"
##df.sort_values("Category", inplace = True)

# Row 10472 removed due to missing value of Category
df.drop(df.index[10472], inplace=True)

# Row [7312,8266] removed due to "Unrated" value in Content Rating
df.drop(df.index[[7312,8266]], inplace=True)

# Replace "NaN" with mean 
##imputer = SimpleImputer(missing_values= np.nan, strategy='mean')
#imputer = SimpleImputer()
#data['Rating'] = imputer.fit_transform(data[['Rating']])
'''

# Rounding the mean value to 1 decimal place
##df['Rating'].round(1)
##df.dropna(axis=0, inplace=True)
#print(df.head())
#df.to_csv('C:\\Users\\Ruudra\\Desktop\\Python Internship\\CleanedData.csv')
def feature1():
    #print('=========================Feature 1=========================')
    #print()
    d={}
    cate=[]
    per=[]
    totalinstalls_sqlobj = curr.execute('SELECT sum(Installs) FROM CleanedData2')
    for tuple in totalinstalls_sqlobj:
         totalinstlls = tuple[0]
         
    categorywiseinstalls = curr.execute('SELECT Category,sum(Installs) FROM CleanedData2 GROUP by Category;')   
    
    for tuple in categorywiseinstalls:
        d[tuple[0]] = round((tuple[1]/totalinstlls)*100,2)
        cate=d.keys()
        per=d.values()
    root=tk.Tk()
    fig = plt.figure(figsize=(6,5), dpi=100)
    ax = figure.add_subplot(111)
    chart_type = FigureCanvasTkAgg(figure, root)
    chart_type.get_tk_widget().pack()
    fig.set_facecolor('grey')
    plt.xticks(rotation=90,c='w')
    d1=d[['tuple[0]','tuple[1]'].groupby()
    d1.plot(kind='pie', legend=True)# ax=ax1)
    #df1 = df1[['Country','GDP_Per_Capita']].groupby('Country').sum()
    ax1.set_title('Feature 1')
    '''
    plt.xlabel("Categories",c='black')
    plt.ylabel("Percentage Downloads(%)",c='black')
    plt.yticks(c='w')
    plt.bar(cate,per,color='b')
    '''
    #plt.title("Feature 1",c='w')
    #plt.show()
    root.mainloop()
    
def feature2():
    print('========================Feature 2============================')
    print()
    print("Enter two No of Installs")
    x = int(input())
    y = int(input())
    a = (x,y)
    count=curr.execute('SELECT count(App) FROM CleanedData2 where  Installs > ? and Installs <= ?',a)    
    for tuple in count:
        cases=tuple[0]
        print("No of apps :",cases)
        '''
    plt.bar(a,cases)
    plt.xlabel('Downloads')
    plt.ylabel('Apps')
    plt.title("Feature 2")
    plt.show()
    '''
    
def feature3():
    print('========================Feature 3============================')
    maxcategory_sqlobj = curr.execute('SELECT Category,max(AvgInstalls) FROM (SELECT Category,avg(installs) as AvgInstalls FROM CleanedData2 GROUP by Category)')
    for tuple in maxcategory_sqlobj:
        maxcategory = tuple[0]
        maxavginstalls = tuple[1]
    print("Category with most downloads:",maxcategory,maxavginstalls)
    mincategory_sqlobj = curr.execute('SELECT Category,min(AvgInstalls) FROM (SELECT Category,avg(installs) as AvgInstalls FROM CleanedData2 GROUP by Category)')
    for tuple in mincategory_sqlobj:
        mincategory = tuple[0]
        minavginstalls = tuple[1]
    print("Category of apps with least downloads:",mincategory,minavginstalls)
    #cate=[]
    print()
    avgin=[]
    above250k = []
    above250k_sqlobj = curr.execute('SELECT Category,AvgInstalls FROM (SELECT Category,avg(installs) as AvgInstalls FROM CleanedData2 GROUP by Category ) WHERE AvgInstalls >= 250000')
    for tuple in above250k_sqlobj:
        above250k.append(tuple[0])
        avgin.append(tuple[1])
    #print(avgin)
    print("Apps with average of 250000 downloads are:")
    fig = plt.figure()
    fig.set_facecolor('grey')
    plt.xticks(rotation=90,c='w')
    plt.yscale('log')
    plt.xlabel("Categories",c='black')
    plt.ylabel("Average Downloads",c='black')
    plt.bar(above250k,avgin,color='g')
    #plt.pie(avgin,labels=above250k,radius=2.5,autopct='%0.2f%%', rotatelabels=True)
    plt.title("Feature 3",c='w')
    plt.show()

def feature4():
    print('========================Feature 4============================')
    d = {}
    maxcategory_sqlobj = curr.execute('SELECT Category,max(AvgRating) FROM (SELECT Category,avg(Rating) as AvgRating FROM CleanedData2 GROUP by Category)')
    for tuple in maxcategory_sqlobj:
        maxcategory = tuple[0]
    print("Category with max avg ratings is",maxcategory)

    avgratings_table_sqlobj = curr.execute('SELECT Category,avg(Rating) as AvgRating FROM CleanedData2 GROUP by Category')
    for tuple in avgratings_table_sqlobj:
        d[tuple[0]] = round(tuple[1],5)
    #print (d)
    fig = plt.figure()
    fig.set_facecolor('grey')
    values = d.values()
    labels = d.keys()
    plt.xticks(rotation=90)
    plt.xlabel("Categories")
    plt.ylabel("Average Maximum Ratings")
    plt.scatter(labels,values)
    plt.show()

def feature5():
    print('========================Feature 5============================')
    print("Enter two size numbers")
    x = int(input())
    y = int(input())
    a = (x,y)
    avginstalls_apps_forgivenrange_sqlobj = curr.execute('SELECT sum(Installs) FROM CleanedData WHERE Size>=? AND Size<=?',a)
    for tuple in avginstalls_apps_forgivenrange_sqlobj:
        suminstalls_apps_in_range = tuple[0]
    print("Number of Installs:",suminstalls_apps_in_range)
    #plt.bar(a,suminstalls_apps_in_range)
    #plt.xlabel('Size')
    #plt.ylabel('Downloads')
    #plt.yscale('log')
    #plt.title("Feature 5")
    #plt.show()


def feature6():
    print('========================Feature 6============================')
    a=input("Enter Any Year between 2015 and 2019 :")
    b=("%"+a,)
    suminstallsmin=curr.execute("SELECT Category,min(InstallCount) FROM (SELECT Category,sum(Installs) as InstallCount FROM CleanedData WHERE LastUpdated like ? group by Category)",b)
    for tuple in suminstallsmin:
        cases1=tuple
        print("Least Downloads=",cases1)
    suminstallsmax=curr.execute("SELECT Category,max(InstallCount) FROM (SELECT Category,sum(Installs) as InstallCount FROM CleanedData WHERE LastUpdated like ? group by Category)",b)
    for tuple in suminstallsmax:
        cases2=tuple
        print("Most Downloads=",cases2)
    

def feature7():
    print('========================Feature 7============================')
    d_increase = {}
    d_decrease = {}
    avginstalls_sqlobj = curr.execute('SELECT avg(Installs) FROM CleanedData2')
    for tuple in avginstalls_sqlobj:
         avginstalls = tuple[0]
    a = ('Varies with device',)
    androidver_noissue_apps_sqlobj = curr.execute('SELECT App,Installs FROM CleanedData2 WHERE AndroidVer = ?', a)
    for tuple in androidver_noissue_apps_sqlobj:
        if tuple[1] > avginstalls:
            d_increase[tuple[0]] = round(((tuple[1]-avginstalls)/avginstalls)*100,2)
        else:
            d_decrease[tuple[0]] = round(((avginstalls-tuple[1])/avginstalls)*100,2)
    #print (d_increase)
    xi=d_increase.values()
    yi=d_increase.keys()
    print()
    #print (d_decrease)
    xd=d_decrease.values()
    yd=d_decrease.keys()
    print()
    androidver_noissue_avginstalls_sqlobj = curr.execute('SELECT avg(Installs) FROM CleanedData2 WHERE AndroidVer = ?', a)
    for tuple in androidver_noissue_avginstalls_sqlobj:
        androidver_noissue_avginstalls = tuple[0]
    print(androidver_noissue_avginstalls)
    if androidver_noissue_avginstalls > avginstalls:
        percentincrease = round(((androidver_noissue_avginstalls-avginstalls)/avginstalls)*100, 2)
        print("Percent increase =",percentincrease)
    else:
        percentdecrease = round(((avginstalls-androidver_noissue_avginstalls)/avginstalls) *100,2)
        print("Percent decrease =",percentdecrease)

    fig = plt.figure()
    fig.set_facecolor('grey')
    g=fig.add_axes([1,1,5,5])
    plt.xticks(rotation=90,c='w')
    plt.yticks(c='w')
    x=yi
    y=xi
    g.scatter(x,y,color='g')
    g.set_title("Percentage Increase",c='w')
    plt.xticks(rotation=90,c='w')
    plt.xticks(rotation=90,c='w')
    a=yd
    b=xd
    g.scatter(a,b,color='r')
    plt.title("Percentage Decrease",c='w')
    
    plt.show()
    
def feature8():
    test=[]
    for i in range(5):
        a = random.choice(categories)
        while a in test:
            a = random.choice(categories)
        test.append(a)
    #print(test)
    cat_analysis = {}
    for i in test:
        cat_analysis_sqlobj = curr.execute('SELECT sum(Installs),avg(Rating),avg(size) FROM CleanedData2 WHERE Category = ?',(i,))
        for j in cat_analysis_sqlobj:
            item = j
        t = list(item + (0,))
        cat_analysis[i] = t
    #print(cat_analysis)
    l = list(cat_analysis.keys())
    installs = []
    ratings = []
    size = []
    pts = []
    for i in l:
        installs.append(cat_analysis[i][0])
        ratings.append(cat_analysis[i][1])
        size.append(cat_analysis[i][2])
    p1 = installs.index(max(installs))
    p2 = ratings.index(max(ratings))
    p3 = size.index(min(size))
    cat_analysis[l[p1]][3] += 1
    cat_analysis[l[p2]][3] += 1
    cat_analysis[l[p3]][3] += 1
    for i in l:
        pts.append(cat_analysis[i][3])
    m = pts.index(max(pts))
    print("Category with most likely download is :",l[m])
    for i in l:
        tup = (i, cat_analysis[i][1],cat_analysis[i][0])
        curr.execute('INSERT INTO CategoryAnalysis (Category, Rating, Installs) VALUES (?, ?, ?)', tup)
'''   
def feature9():
    print('========================Feature 9============================')    
    apprateandinst=curr.execute("SELECT App FROM CleanedData WHERE Installs>=100000 and Rating >=4.1")
    for tuple in apprateandinst:
        appnrate=tuple
        print(appnrate)
'''
def feature9():
    apprateandinst = curr.execute("SELECT count(App) FROM CleanedData WHERE Installs >= 100000")
    for tuple in apprateandinst:
        allapps = tuple[0]
    apprateandinst = curr.execute("SELECT count(App) FROM CleanedData WHERE Installs>=100000 and Rating >=4.1")
    for tuple in apprateandinst:
        appnrate = tuple[0]
    if allapps > appnrate:
        percent = round((appnrate)/allapps *100, 2)
        print(percent, "% of the apps with installs greater than 1mil have a rating greater than 4")
    else:
        print("All the apps with installs greater than 1mil have a rating greater than 4")
    

def feature10():
    print('========================Feature 10============================')
    dict_maxmonth = {}
    months = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December']
    categories = curr.execute('SELECT Category FROM CleanedData2')
    for category in categories:
        maxinstalls = 0
        for month in months:
            a = (category, month+"%")
            installs_monthwise_sqlobj = curr.execute('SELECT Category,sum(Installs) as SumOfInstalls FROM CleanedData2 WHERE Category = ? AND LastUpdated like ?', a)
            for tuple in installs_monthwise_sqlobj:
                if type(tuple[1]) == int:
                    if tuple[1] > maxinstalls:
                        dict_maxmonth [category] = month
    print (dict_maxmonth)
    a = ('Teen',)
    count_apps_for_teens_sqlobj = curr.execute('SELECT count(App) FROM CleanedData2 WHERE ContentRating = ?', a)
    for tuple in count_apps_for_teens_sqlobj:
        teen = tuple[0]
    b = ('Mature 17+',)
    count_apps_for_mature_sqlobj = curr.execute('SELECT count(App) FROM CleanedData2 WHERE ContentRating = ?', b)
    for tuple in count_apps_for_mature_sqlobj:
        mature = tuple[0]
    ratio = teen/mature
    print (teen,mature,ratio)

def feature11():
    q1=['January', 'February', 'March']
    q2=['April', 'May', 'June']
    q3=['July', 'August', 'September']
    q4=['October', 'November', 'December']
    y = []
    for i in range(2010,2019):
        s = 0
        for j in q1:
            date = j + "%" + str(i)
            sum_installs_sqlobj = curr.execute("SELECT sum(Installs) as SumOfInstalls FROM CleanedData WHERE LastUpdated like ?",(date,))
            for tuple in sum_installs_sqlobj:
                if tuple[0] == None:
                    s += 0
                else:
                    s += tuple[0]
        y.append(s)
        s = 0
        for j in q2:
            date = j + "%" + str(i)
            sum_installs_sqlobj = curr.execute("SELECT sum(Installs) as SumOfInstalls FROM CleanedData WHERE LastUpdated like ?",(date,))
            for tuple in sum_installs_sqlobj:
                if tuple[0] == None:
                    s = s + 0
                else:
                    s = s + tuple[0]
        y.append(s)
        s = 0
        for j in q3:
            date = j + "%" + str(i)
            sum_installs_sqlobj = curr.execute("SELECT sum(Installs) as SumOfInstalls FROM CleanedData WHERE LastUpdated like ?",(date,))
            for tuple in sum_installs_sqlobj:
                if tuple[0] == None:
                    s += 0
                else:
                    s += tuple[0]
        y.append(s)
        s = 0
        for j in q4:
            date = j + "%" + str(i)
            sum_installs_sqlobj = curr.execute("SELECT sum(Installs) as SumOfInstalls FROM CleanedData WHERE LastUpdated like ?",(date,))
            for tuple in sum_installs_sqlobj:
                if tuple[0] == None:
                    s += 0
                else:
                    s += tuple[0]
        y.append(s)
    i = y.index(max(y))
    q = i%4
    year = i//4
    year = '201' + str(year)
    if q==0:
        print("Max downloads in year",year," and quarter 1")
    elif q==1:
        print("Max downloads in year",year," and quarter 2")
    elif q==2:
        print("Max downloads in year",year," and quarter 3")
    else:
        print("Max downloads in year",year," and quarter 4")

def feature12():
    print('========================Feature 12============================')    
    positivemax=curr.execute("SELECT App,max(positivecount) FROM (SELECT App,count(App) as positivecount FROM googleplaystore_user_reviews WHERE Sentiment = 'Positive' GROUP by App)")     
    for tuple in positivemax:
        maxp=tuple
        print("Most Positive Sentiments=",maxp)
    negativemax=curr.execute("SELECT App,max(positivecount) FROM (SELECT App,count(App) as positivecount FROM googleplaystore_user_reviews WHERE Sentiment = 'Negative' GROUP by App)")     
    
    for tuple in negativemax:
        maxn=tuple
        print("Most Negative Sentiments=",maxn)
    #found=0
    plusratio={}
    minusratio={}
    positiveratios=("SELECT App,count(App) as positivecount FROM googleplaystore_user_reviews WHERE Sentiment = 'Positive' GROUP by App")
    for tuple in positiveratios:
        plusratio[tuple[0]]=tuple[1]
        #print(plusratio)
    negativeratios=("SELECT App,count(App) as negativecount FROM googleplaystore_user_reviews WHERE Sentiment = 'Negative' GROUP by App")
    for tuple in negativeratios:
        minusratio[tuple[0]]=tuple[1]
        #print(minusratio)

def feature14():
    print('========================Feature 14============================')
    x=input("Enter App Name:")
    y=input("Enter Sentiment:")
    a=(x,y)
    sentimental=curr.execute("SELECT Translated_Review FROM googleplaystore_user_reviews WHERE App = ? AND Sentiment = ? order by Sentiment_Polarity DESC",a)
    for tuple in sentimental:
        reviewsofapps=tuple[0]
        print(reviewsofapps)

def feature15():
    d = {}
    apps = []
    avg_sentimentpolarity_sqlobj = curr.execute('SELECT App,avg(Sentiment_Polarity) as AvgSentimentPolarity FROM googleplaystore_user_reviews GROUP by App ORDER by AvgSentimentPolarity DESC')
    for tuple in avg_sentimentpolarity_sqlobj:
        d[tuple[0]] = tuple[1]
        apps.append(tuple[0])
    i = 0
    l = len(apps)
    app = input("Select an app : ")
    i = apps.index(app)
    p = round((l-i)/l *100, 2)
    print("The selected app is better than", p, "% of apps.")
    if p>=60:
        print("It is advisable to launch a similar app")
    else:
        print("It is not advisable to launch a similar app")

def feature16():
    print('========================Feature 16============================')
    months = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December']
    print("Enter Year:")
    yr=int(input())
    l=[]
    for month in months:
        a=(month+"%"+str(yr))
        avginstyearwise=curr.execute("SELECT avg(Installs) as avg_of_installs FROM CleanedData2 WHERE LastUpdated like ?",(a,)) 
        for tuple in avginstyearwise:
            if tuple[0]==None:
                l.append(0)
            else:    
                l.append(tuple[0]) 
        m=months[l.index(max(l))]
    print("Best month for ",yr,"is",m)
            

def feature17():
    print('========================Feature 17============================')        
    avginstalls=curr.execute("SELECT avg(Installs) FROM CleanedData ")
    for tuple in avginstalls:
        avinst=tuple[0]
    avgsize=curr.execute("SELECT avg(Size) FROM CleanedData ")
    for tuple in avgsize:
        avsize=tuple[0]
    a=(avsize,avinst)
    positivetrend=curr.execute("SELECT count(App) FROM CleanedData WHERE Size>= ? AND Installs >= ?",a)
    for tuple in positivetrend:
        print(tuple)
    negativetrend=curr.execute("SELECT count(App) FROM CleanedData WHERE Size>= ? AND Installs <= ?",a)
    for tuple in negativetrend:
        print(tuple)
    print("Yes,the trend is negative with the increase in the app size.")

        


feature1() 
#feature2() 
#feature3() 
#feature4() 
#feature5() 
#feature6() 
#feature7() #gui too big for consonle
#feature8() 
#feature9() ##correlation baaki
#feature10() #error : InterfaceError: Error binding parameter 0 - probably unsupported type.
#feature11() 
#feature12()
#feature13() #Learn Regression
#feature14() 
#feature15() #error
#feature16() 
#feature17() #correlation try
#feature18()
#feature19()
#feature20)()