import pandas as pd
import numpy as np
import sqlite3
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression

#Connecting to Database - FinalData.db
con = sqlite3.connect('/Users/ruudrapatel/Desktop/Study/PythonProgs/PyInterns/FinalData.db', isolation_level = None)
curr = con.cursor()

#print("To RUN this app type this in terminal : streamlit run /Users/ruudrapatel/Desktop/Study/PythonProgs/PyInterns/project_final.py")

#Cleaning App Data and ReviewsTable
df = pd.read_csv("/Users/ruudrapatel/Desktop/Study/PythonProgs/PyInterns/googleplaystore-App-data.csv")
df2 = pd.read_csv("/Users/ruudrapatel/Desktop/Study/PythonProgs/PyInterns/googleplaystore_user_reviews.csv")
df['Installs']= df['Installs'].str.replace('+','')
df['Installs'] =df['Installs'].str.replace(',','')
df['Size'] = df['Size'].apply(lambda x: str(x).replace('Varies with device', '') if 'Varies with device' in str(x) else x)
df['Size'] = df['Size'].apply(lambda x: str(x).replace('M', '') if 'M' in str(x) else x)
df['Size'] = df['Size'].apply(lambda x: str(x).replace(',', '') if 'M' in str(x) else x)
df['Size'] = df['Size'].apply(lambda x: float(str(x).replace('k', '')) / 1000 if 'k' in str(x) else x)
#df.dropna(how='any',inplace=True)
total  = df.isnull().sum().sort_values(ascending=False)
percent =  (df.isnull().sum() / df.isnull().count()).sort_values(ascending=False)
missing_data  = pd.concat([total,percent],axis=1,keys=['Total','Percent'])
#print(missing_data.head())

data_free  = df[df['Type'] == 'Free']
data_paid =  df[df['Type'] == 'Paid']

#df2.dropna(how='any',inplace=True)
#print(df2.shape)

#Saving CleanedData and ReviewsTable

#df.to_csv("/Users/ruudrapatel/Desktop/Study/PythonProgs/PyInterns/CleanedData.csv")
#df2.to_csv("/Users/ruudrapatel/Desktop/Study/PythonProgs/PyInterns/ReviewsTable.csv")

categories = []
categories_sqlobj = curr.execute('SELECT Category FROM CleanedData group by Category')
for tuple in categories_sqlobj:
    categories.append(tuple[0])

def feature1():
    d={}
    totalinstalls_sqlobj = curr.execute('SELECT sum(Installs) FROM CleanedData')
    for tuple in totalinstalls_sqlobj:
        totalinstlls = tuple[0]
    categorywiseinstalls = curr.execute('SELECT Category,sum(Installs) FROM CleanedData GROUP by Category;')
    for tuple in categorywiseinstalls:
        d[tuple[0]] = round((tuple[1]/totalinstlls)*100,2)
    labels = list(d.keys())
    y = np.array(list(d.values()))
    fig = px.pie(names = labels, values = y, title = 'Category wise installs',width=900,height=700)
    st.plotly_chart(fig)


def feature2():
    labels = []
    values = []
    print("No of apps download wise :-")
    x = 10000
    y = 50000
    labels.append(str(x)+'-'+str(y))
    a = (x,y)
    count=curr.execute('SELECT count(App) FROM CleanedData where  Installs > ? and Installs <= ?',a)    
    for tuple in count:
        cases=tuple[0]
        values.append(cases)
        print("No of apps with downloads in range of ", x, "and", y, ":",cases)
    x = 50000
    y = 150000
    labels.append(str(x)+'-'+str(y))
    a = (x,y)
    count=curr.execute('SELECT count(App) FROM CleanedData where  Installs > ? and Installs <= ?',a)    
    for tuple in count:
        cases=tuple[0]
        values.append(cases)
        print("No of apps with downloads in range of ", x, "and", y, ":",cases)
    x = 150000
    y = 500000
    labels.append(str(x)+'-'+str(y))
    a = (x,y)
    count=curr.execute('SELECT count(App) FROM CleanedData where  Installs > ? and Installs <= ?',a)    
    for tuple in count:
        cases=tuple[0]
        values.append(cases)
        print("No of apps with downloads in range of ", x, "and", y, ":",cases)
    x = 500000
    y = 5000000
    labels.append(str(x)+'-'+str(y))
    a = (x,y)
    count=curr.execute('SELECT count(App) FROM CleanedData where  Installs > ? and Installs <= ?',a)    
    for tuple in count:
        cases=tuple[0]
        values.append(cases)
        print("No of apps with downloads in range of ", x, "and", y, ":",cases)
    x = 5000000
    labels.append('>'+str(x))
    a = (x,)
    count=curr.execute('SELECT count(App) FROM CleanedData where  Installs >= ?',a)    
    for tuple in count:
        cases=tuple[0]
        values.append(cases)
        print("No of apps with downloads greater than ", x, ":",cases)
    fig = go.Figure([go.Bar(x=labels, y=values, text=values, textposition='auto')])
    fig.update_layout(xaxis_title="Downloads (Groupwise)",yaxis_title="Number of Apps")
    st.plotly_chart(fig)


def feature3():
    maxcategory_sqlobj = curr.execute('SELECT Category,max(AvgInstalls) FROM (SELECT Category,avg(installs) as AvgInstalls FROM CleanedData GROUP by Category)')
    for tuple in maxcategory_sqlobj:
        maxcategory = tuple[0]
        maxavginstalls = tuple[1]
    print(maxcategory,maxavginstalls)
    
    mincategory_sqlobj = curr.execute('SELECT Category,min(AvgInstalls) FROM (SELECT Category,avg(installs) as AvgInstalls FROM CleanedData GROUP by Category)')
    for tuple in mincategory_sqlobj:
        mincategory = tuple[0]
        minavginstalls = tuple[1]
    print(mincategory,minavginstalls)
    
    d={}
    categorywiseinstalls = curr.execute('SELECT Category,sum(Installs) FROM CleanedData GROUP by Category;')
    for tuple in categorywiseinstalls:
        d[tuple[0]] = tuple[1]
    
    colors = ['lightblue']*len(categories)
    values = list(d.values())
    labels = list(d.keys())
    maxinstalls = max(values)
    maxindex = values.index(maxinstalls)
    mininstalls = min(values)
    minindex = values.index(mininstalls)
    for i,x in enumerate(values):
        if x>250000:
            colors[i] = 'yellow'
    colors[minindex] = 'red'
    colors[maxindex] = 'green'

    fig = go.Figure([go.Bar(x=labels, y=values, marker_color = colors)])
    fig.update_layout(xaxis_title = "Categories",yaxis_title = "Downloads")
    st.plotly_chart(fig)
    st.write('Category with minimum installs :- ',labels[minindex],' (',mininstalls,')')
    st.write('Category with maximum installs :- ',labels[maxindex],' (',maxinstalls,')')


def feature4():
    d = {}
    avgratings_table_sqlobj = curr.execute('SELECT Category,avg(Rating) as AvgRating FROM CleanedData GROUP by Category')
    for tuple in avgratings_table_sqlobj:
        d[tuple[0]] = round(tuple[1],5)
    labels = list(d.keys())
    values = list(d.values())
    colors = ['lightblue']*len(categories)
    colors[values.index(max(values))] = 'green'
    colors[values.index(min(values))] = 'red'
    fig = go.Figure([go.Bar(x=labels, y=values, marker_color = colors)])
    fig.update_layout(xaxis_title = "Categories",yaxis_title = "Rating")
    st.plotly_chart(fig)


def feature5():
    values = []
    labels = []
    x = 10
    y = 20
    a = (x,y)
    labels.append(str(x)+'-'+str(y))
    avginstalls_sizewise_sqlobj = curr.execute('SELECT sum(Installs) FROM CleanedData WHERE Size>=? AND Size<=?',a)
    for tuple in avginstalls_sizewise_sqlobj:
        values.append(tuple[0])
    x = 20
    y = 30
    a = (x,y)
    labels.append(str(x)+'-'+str(y))
    avginstalls_sizewise_sqlobj = curr.execute('SELECT sum(Installs) FROM CleanedData WHERE Size>=? AND Size<=?',a)
    for tuple in avginstalls_sizewise_sqlobj:
        values.append(tuple[0])
    x = 30
    a = (x,)
    labels.append('>'+str(x))
    avginstalls_sizewise_sqlobj = curr.execute('SELECT sum(Installs) FROM CleanedData WHERE Size>=?',a)
    for tuple in avginstalls_sizewise_sqlobj:
        values.append(tuple[0])
    fig = go.Figure([go.Bar(x=labels, y=values)])
    fig.update_layout(xaxis_title = "Size",yaxis_title = "Downloads")
    st.plotly_chart(fig)


def feature6():
    d = {}
    a=2016
    b=("%"+str(a),)
    suminstall_yearwise = curr.execute("SELECT Category,sum(Installs) as InstallCount FROM CleanedData WHERE LastUpdated like ? group by Category",b)
    for tuple in suminstall_yearwise:
        d[tuple[0]] = tuple[1]
    labels = list(d.keys())
    values = list(d.values())
    
    colors = ['lightblue']*len(categories)
    colors[values.index(max(values))] = 'green'
    colors[values.index(min(values))] = 'red'
    fig = go.Figure([go.Bar(x=labels, y=values, marker_color = colors)])
    st.write('For year ', str(a),' :-')
    fig.update_layout(xaxis_title = "Categories",yaxis_title = "Downloads")
    st.plotly_chart(fig)
    st.write('Category with most installs :- ',labels[values.index(max(values))],'(',max(values),')')
    st.write('Category with least installs :- ',labels[values.index(min(values))],'(',min(values),')')

    d = {}
    a=2017
    b=("%"+str(a),)
    suminstall_yearwise = curr.execute("SELECT Category,sum(Installs) as InstallCount FROM CleanedData WHERE LastUpdated like ? group by Category",b)
    for tuple in suminstall_yearwise:
        d[tuple[0]] = tuple[1]
    labels = list(d.keys())
    values = list(d.values())
    colors = ['lightblue']*len(categories)
    colors[values.index(max(values))] = 'green'
    colors[values.index(min(values))] = 'red'
    fig = go.Figure([go.Bar(x=labels, y=values, marker_color = colors)])
    st.write('For year ', str(a),' :-')
    st.plotly_chart(fig)
    st.write('Category with most installs :- ',labels[values.index(max(values))],'(',max(values),')')
    st.write('Category with least installs :- ',labels[values.index(min(values))],'(',min(values),')')

    d = {}
    a=2018
    b=("%"+str(a),)
    suminstall_yearwise = curr.execute("SELECT Category,sum(Installs) as InstallCount FROM CleanedData WHERE LastUpdated like ? group by Category",b)
    for tuple in suminstall_yearwise:
        d[tuple[0]] = tuple[1]
    labels = list(d.keys())
    values = list(d.values())
    colors = ['lightblue']*len(categories)
    colors[values.index(max(values))] = 'green'
    colors[values.index(min(values))] = 'red'
    fig = go.Figure([go.Bar(x=labels, y=values, marker_color = colors)])
    st.write('For year ', str(a),' :-')
    st.plotly_chart(fig)
    st.write('Category with most installs :- ',labels[values.index(max(values))],'(',max(values),')')
    st.write('Category with least installs :- ',labels[values.index(min(values))],'(',min(values),')')


def feature7():
    labels = [i for i in range(2010,2019)]
    #print(labels)
    values = []
    for i,x in enumerate(labels):
        t = ('Varies with device', '%'+str(x))
        s = ('%'+str(x),)
        vernoissue_yearwise_installs_sqlobj = curr.execute('SELECT count(Installs) FROM CleanedData WHERE AndroidVer = ? AND LastUpdated like ?', t)
        for tuple in vernoissue_yearwise_installs_sqlobj:
            a = tuple[0]
        yearwise_installs_sqlobj = curr.execute('SELECT count(Installs) FROM CleanedData WHERE LastUpdated like ?', s)
        for tuple in yearwise_installs_sqlobj:
            b = tuple[0]
        values.append(round((a/b)*100, 5))
    fig = px.line(x=labels, y=values)
    fig.update_layout(xaxis_title="Downloads",yaxis_title="Years")
    st.plotly_chart(fig)
    st.write('The above graph shows us the % of apps whose android version is not an issue over the years.')


def feature8():
    slopes = []
    y = []
    categories_given = ['SPORTS', 'ENTERTAINMENT', 'SOCIAL', 'NEWS_AND_MAGAZINES', 'EVENTS', 'TRAVEL_AND_LOCAL', 'GAME']
    years = [i for i in range(2010,2019)]
    curr.execute("""CREATE TABLE if NOT EXISTS "Feature8_table" ("Category" TEXT Primary key,"Installs" INTEGER);""")
    for category in categories_given:
        l = []
        for year in years:
            b=(category,"%"+str(year))
            installs_sqlobj = curr.execute("SELECT avg(Installs) FROM CleanedData WHERE Category = ? AND LastUpdated like ?;", b)
            for tuple in installs_sqlobj:
                if tuple[0] == None:
                    l.append(0)
                else:
                    l.append(tuple[0])
        years_np = np.array(years)
        years_np = years_np.reshape(-1,1)
        model = LinearRegression()
        model.fit(years_np,l)
        coef = model.coef_
        slopes.append(coef[0])
        y_range = model.predict(years_np)
        trace = go.Scatter(x=years,y=y_range,name=category)
        y.append(trace)
    layout = {'title':'Average installs vs years',
             'xaxis':{'title':'Years'},'yaxis':{'title':'Average installs'}}
    fig = go.Figure(data=y,layout=layout)
    st.plotly_chart(fig)
    best_cat = categories_given[slopes.index(max(slopes))]
    st.write('The category of app to be most likely downloaded will be', best_cat,'. This can be seen by the best fit plot for it.')
    for category in categories_given:
        a = (category,)
        categorywiseinstalls_sqlobj = curr.execute('SELECT sum(Installs) FROM CleanedData WHERE Category = ?;',a)
        for tuple in categorywiseinstalls_sqlobj:
            b = (category, tuple[0])
        querry = "INSERT OR REPLACE INTO Feature8_table (Category, Installs) VALUES ('%s', %s)" %b
        curr.execute(querry)
    st.write('Total installs for above category updated in database.')


def feature9():
    apprateandinst = curr.execute("SELECT count(App) FROM CleanedData WHERE Installs >= 100000")
    for tuple in apprateandinst:
        allapps = tuple[0]
    apprateandinst = curr.execute("SELECT count(App) FROM CleanedData WHERE Installs>=100000 and Rating >=4.1")
    for tuple in apprateandinst:
        appnrate = tuple[0]
    labels = ['<4.1','>=4.1']
    y = []
    a = round((allapps-appnrate)*100/allapps,2)
    y.append(a)
    a = round(appnrate*100/allapps,2)
    y.append(a)
    fig = px.pie(names = labels, values = y, title = 'Apps with installs more than 100000')
    st.plotly_chart(fig)
    if allapps > appnrate:
        st.write(a, "% of the apps with installs greater than 100k have a rating greater than 4.1")
    else:
        st.write("All the apps with installs greater than 100k have a rating greater than 4.1")
    installs = []
    ratings = []
    installs_size_sqlobj = curr.execute("SELECT Rating,Installs FROM CleanedData")
    for tuple in installs_size_sqlobj:
        ratings.append(tuple[0])
        installs.append(tuple[1])
    ratings_np = np.array(ratings)
    ratings_np = ratings_np.reshape(-1,1)
    model = LinearRegression()
    model.fit(ratings_np,installs)
    x_range = np.linspace(ratings_np.min(),ratings_np.max(),100)
    y_range = model.predict(x_range.reshape(-1,1))
    fig = px.scatter(x=ratings, y=installs, opacity=0.65)
    fig.add_traces(go.Scatter(x=x_range, y=y_range, name='Regression Fit'))
    fig.update_layout(xaxis_title="Rating",yaxis_title="Downloads (In Billions")
    st.plotly_chart(fig)
    coef = model.coef_
    m = coef[0]
    if m>0:
        st.write('As ratings increase, number of installs increase which is indicated by the slope of the best fit line.')
    elif m<0:
        st.write('As ratings increase, number of installs decrease which is indicated by the slope of the best fit line.')
    else:
        st.write('The ratings and number of installs are not independent.')


def feature10():
    dict_maxmonth = {}
    months = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December']
    for category in categories:
        maxinstalls = 0
        for month in months:
            a = (category, month+"%")
            installs_monthwise_sqlobj = curr.execute('SELECT Category,sum(Installs) as SumOfInstalls FROM CleanedData WHERE Category = ? AND LastUpdated like ?', a)
            for tuple in installs_monthwise_sqlobj:
                if type(tuple[1]) == int:
                    if tuple[1] > maxinstalls:
                        dict_maxmonth [category] = month
    print (dict_maxmonth)
    a = ('Teen',)
    labels = ['Teen','Mature 17+']
    values = []
    count_apps_for_teens_sqlobj = curr.execute('SELECT count(App) FROM CleanedData WHERE ContentRating = ?', a)
    for tuple in count_apps_for_teens_sqlobj:
        values.append(tuple[0])
    b = ('Mature 17+',)
    count_apps_for_mature_sqlobj = curr.execute('SELECT count(App) FROM CleanedData WHERE ContentRating = ?', b)
    for tuple in count_apps_for_mature_sqlobj:
        values.append(tuple[0])
    fig = px.pie(names = labels, values = values, title = 'Mature 17+ vs Teen (Installs)')
    st.plotly_chart(fig)
    if values[0]>values[1]:
        ratio = values[0]/values[1]
        st.write('The ratio of Mature 17+ to Teen is 1 :', ratio)
    elif values[1]>values[0]:
        ratio = values[1]/values[0]
        st.write('The ratio of Teen to Mature 17+ is 1 :', ratio)
    else:
        st.write('The ratio of Mature 17+ to Teen is 1 : 1')


def feature11():
    q1=['January', 'February', 'March']
    q2=['April', 'May', 'June']
    q3=['July', 'August', 'September']
    q4=['October', 'November', 'December']
    labels = []
    y = []
    for i in range(2016,2019):
        s = 0
        for j in q1:
            date = j + "%" + str(i)
            sum_installs_sqlobj = curr.execute("SELECT sum(Installs) as SumOfInstalls FROM CleanedData WHERE LastUpdated like ?",(date,))
            for tuple in sum_installs_sqlobj:
                if tuple[0] == None:
                    s += 0
                else:
                    s += tuple[0]
        lbl = 'Quarter 1 (' + str(i) + ')'
        labels.append(lbl)
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
        lbl = 'Quarter 2 (' + str(i) + ')'
        labels.append(lbl)
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
        lbl = 'Quarter 3 (' + str(i) + ')'
        labels.append(lbl)
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
        lbl = 'Quarter 4 (' + str(i) + ')'
        labels.append(lbl)
    i = y.index(max(y))
    q = i%4
    year = i//4
    year = int('201' + str(year))
    colors = ['lightblue']*len(labels)
    colors[i] = 'green'
    fig = go.Figure([go.Bar(x=labels, y=y, marker_color = colors)])
    fig.update_layout(xaxis_title="Quarter of the Year",yaxis_title="Downloads (In Billions)")
    st.plotly_chart(fig)
    st.write('The quarter with max downloads is :', labels[i])


def feature12():
    positivemax_sqlobj = curr.execute("SELECT App,max(positivecount) FROM (SELECT App,count(App) as positivecount FROM ReviewsTable WHERE Sentiment = 'Positive' GROUP by App)")
    for tuple in positivemax_sqlobj:
        positivemax_app = tuple[0]
        st.write("The app with most Positive Sentiments :- ", positivemax_app)
    negativemax_sqlobj = curr.execute("SELECT App,max(negativecount) FROM (SELECT App,count(App) as negativecount FROM ReviewsTable WHERE Sentiment = 'Negative' GROUP by App)")
    for tuple in negativemax_sqlobj:
        negativemax_app = tuple[0]
        st.write("The app with most Negative Sentiments :- ", negativemax_app)
    d_positive_reviewcount_appwise = {}
    d_negative_reviewcount_appwise = {}
    positive_reviews_count_sqlobj = curr.execute("SELECT App,count(App) as positivecount FROM ReviewsTable WHERE Sentiment = 'Positive' GROUP by App ORDER by App ASC")
    for tuple in positive_reviews_count_sqlobj:
        d_positive_reviewcount_appwise[tuple[0]] = tuple[1]
    negative_reviews_count_sqlobj = curr.execute("SELECT App,count(App) as negativecount FROM ReviewsTable WHERE Sentiment = 'Negative' GROUP by App ORDER by App ASC")
    for tuple in negative_reviews_count_sqlobj:
        d_negative_reviewcount_appwise[tuple[0]] = tuple[1]
    apps_list_sqlobj = curr.execute('SELECT App FROM ReviewsTable GROUP by App ORDER by App ASC')
    apps_list = []
    for tuple in apps_list_sqlobj:
        apps_list.append(tuple[0])

    keys=d_positive_reviewcount_appwise.keys()
    for i in apps_list:
        if i in keys:
            continue
        else:
            d_positive_reviewcount_appwise[i] = 0

    keys=d_negative_reviewcount_appwise.keys()
    for i in apps_list:
        if i in keys:
            continue
        else:
            d_negative_reviewcount_appwise[i] = 0
    same_senti_ratioapps = []
    for i in apps_list:
        if d_positive_reviewcount_appwise[i] == d_negative_reviewcount_appwise[i]:
            same_senti_ratioapps.append(i)
    st.write('The apps with same ratio of positive to negative sentiment are :-')
    st.write(same_senti_ratioapps)


def feature13():
    sen_pol = []
    sen_sub = []
    sub_pol_sqlobj = curr.execute("SELECT Sentiment_Polarity,Sentiment_Subjectivity from ReviewsTable")
    for tuple in sub_pol_sqlobj:
        sen_pol.append(tuple[0])
        sen_sub.append(tuple[1])
    sen_pol_np = np.array(sen_pol)
    sen_pol_np = sen_pol_np.reshape(-1,1)
    model = LinearRegression()
    model.fit(sen_pol_np,sen_sub)
    x_range = np.linspace(sen_pol_np.min(),sen_pol_np.max(),100)
    y_range = model.predict(x_range.reshape(-1,1))
    fig = px.scatter(x=sen_pol, y=sen_sub, opacity=0.65 )
    fig.add_traces(go.Scatter(x=x_range, y=y_range, name='Regression Fit'))
    fig.update_layout(xaxis_title = "Sentiment Polarity",yaxis_title = "Sentiment Subjectivity")
    st.plotly_chart(fig)
    coef = model.coef_
    m = coef[0]
    if m>0:
        st.write('As sentiment polarity increases, sentiment subjectivity increases which is indicated by the slope of the best fit line.')
    elif m<0:
        st.write('As sentiment polarity increases, sentiment subjectivity decreases which is indicated by the slope of the best fit line.')
    else:
        st.write('Sentiment polarity does not affect sentiment subjectivity which is indicated by the slope of the best fit line.')
    sen_pol_ip = st.number_input(label='Enter sentiment polarity :-',step=0.2, format="%.2f",min_value=-1.,max_value=1.)
    sen_pol_ip = np.array(sen_pol_ip)
    sen_pol_ip = sen_pol_ip.reshape(1,-1)
    sen_sub_op = model.predict(sen_pol_ip)
    st.write('The predicted sentiment subjectivity for the above value is :', round(sen_sub_op[0], 6))


def feature14():
    all_apps = []
    all_apps_sqlobj = curr.execute("SELECT App FROM ReviewsTable GROUP by App")
    for tuple in all_apps_sqlobj:
        all_apps.append(tuple[0])
    app = st.selectbox('Please select an app :-', all_apps)
    sentiment = st.selectbox("Select category for reviews :-", ['Positive','Negative','Neutral'])
    a=(app,sentiment)
    sentimental=curr.execute("SELECT Translated_Review FROM ReviewsTable WHERE App = ? AND Sentiment = ? order by Sentiment_Polarity DESC",a)
    st.write('The reviews matching the above description are :-')
    i = 1
    for tuple in sentimental:
        st.write(i,')',tuple[0])
        i = i+1


def feature15():
    apps = []
    avg_sentimentpolarity_sqlobj = curr.execute('SELECT App,avg(Sentiment_Polarity) as AvgSentimentPolarity FROM ReviewsTable GROUP by App ORDER by AvgSentimentPolarity DESC')
    for tuple in avg_sentimentpolarity_sqlobj:
        apps.append(tuple[0])
    i = 0
    l = len(apps)
    all_apps = []
    all_apps_sqlobj = curr.execute("SELECT App FROM ReviewsTable GROUP by App")
    for tuple in all_apps_sqlobj:
        all_apps.append(tuple[0])
    app = st.selectbox('Please select an app :-', all_apps)
    i = apps.index(app)
    p_worse = round((l-i)/l *100, 2)
    p_better = 100-p_worse
    y = np.array([p_worse,p_better])
    labels = ['worse than selected', 'better than selected']
    fig = px.pie(names = labels, values = y, title = 'Comparing selected app to others based on sentiment polarity')
    st.plotly_chart(fig)
    if p_worse>= 65:
        st.write('Taking the treshold value to be 65% we can say, it is advisable to launch a similar app.')
    else:
        st.write('Taking the treshold value to be 65% we can say, it is advisable not to launch a similar app.')


def feature16():
    months = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December']
    yr = 2010
    yr = st.number_input('Select a year :',value=2010,min_value=2010,max_value=2018)
    l = []
    for month in months:
        a=(month+"%"+str(yr))
        avginstyearwise=curr.execute("SELECT avg(Installs) as avg_of_installs FROM CleanedData WHERE LastUpdated like ?",(a,)) 
        for tuple in avginstyearwise:
            if tuple[0]==None:
                l.append(0)
            else:
                l.append(tuple[0]) 
        m=months[l.index(max(l))]
    colors = ['lightblue']*len(months)
    colors[months.index(m)] = 'green'
    fig = go.Figure([go.Bar(x=months, y=l, marker_color = colors)])
    fig.update_layout(xaxis_title="Months",yaxis_title="Downloads")
    st.plotly_chart(fig)
    st.write("Best month for",yr,"is",m)


def feature17():
    installs = []
    size = []
    installs_size_sqlobj = curr.execute("SELECT Size,Installs FROM CleanedData where Size != 'Varies with device'")
    for tuple in installs_size_sqlobj:
        size.append(tuple[0])
        installs.append(tuple[1])
    size_np = np.array(size)
    size_np = size_np.reshape(-1,1)
    model = LinearRegression()
    model.fit(size_np,installs)
    x_range = np.linspace(size_np.min(),size_np.max(),100)
    y_range = model.predict(x_range.reshape(-1,1))
    fig = px.scatter(x=size, y=installs, opacity=0.65)
    fig.add_traces(go.Scatter(x=x_range, y=y_range, name='Regression Fit'))
    fig.update_layout(xaxis_title="Size",yaxis_title="Downloads")
    st.plotly_chart(fig)
    coef = model.coef_
    m = coef[0]
    if m>0:
        st.write('Yes, size affects installs. As size increases, number of installs increase which is indicated by the slope of the best fit line.')
    elif m<0:
        st.write('Yes, size affects installs. As size increases, number of installs decrease which is indicated by the slope of the best fit line.')
    else:
        st.write('No, size does not affect number of installs which is indicated by the slope of the best fit line.')


def feature18():
    global df,review_df
    df = pd.read_csv("/Users/ruudrapatel/Desktop/Study/PythonProgs/PyInterns/googleplaystore-App-data.csv")
    review_df = pd.read_csv("/Users/ruudrapatel/Desktop/Study/PythonProgs/PyInterns/ReviewsTable.csv")
    global appEntry,catEntry,ratEntry,reEntry,sizeEntry,insEntry,typeEntry,priceEntry,conEntry,gEntry,luEntry,cvEntry,avEntry
    global appE,trEntry,sEntry,spEntry,ssEntry
    
    col1, col2 = st.beta_columns(2)
    
    list1 = ['Free','Paid']
    list2 = ['Adults only 18+','Everyone','Everyone 10+','Mature 17+','Teen','Unrated']
    list3 = ['Positive', 'Negative', 'Neutral']
    
    with col1:
        st.header("Enter App Data")
        with st.form(key="Form1"):
            appEntry = st.text_input(label="Enter App Name")
            catEntry = st.selectbox('Enter a Category', categories)
            ratEntry = st.number_input(label="Rating",min_value=0.0)
            reEntry = st.number_input(label="Reviews",value=int())
            sizeEntry = st.number_input(label="Size",value=int())
            insEntry = st.number_input(label="Installs",value=int())
            typeEntry = st.selectbox('Enter Type',list1)
            priceEntry = st.number_input(label="Price",min_value=0)
            conEntry = st.selectbox('Enter Content Rating', list2)
            gEntry = st.text_input(label="Generes")
            luEntry = st.text_input(label="Last Updated")
            cvEntry = st.text_input(label="Current Ver")
            avEntry = st.text_input(label="Android Ver")
            if st.form_submit_button(label="Submit App"):
                if appEntry != "":
                    st.success("App Added: "+appEntry)
                    #st.write(appEntry,catEntry,ratEntry,reEntry,sizeEntry,insEntry,typeEntry,priceEntry,conEntry,gEntry,luEntry,cvEntry,avEntry)
                    appdetails = {
                    'App': appEntry,
                    'Category': catEntry,
                    'Rating': ratEntry, 
                    'Reviews': reEntry,
                    'Size': sizeEntry, 
                    'Installs': insEntry, 
                    'Type': typeEntry, 
                    'Price': priceEntry, 
                    'Content Rating': conEntry,
                    'Genres': gEntry,
                    'Last Updated': luEntry, 
                    'Current Ver': cvEntry,
                    'Android Ver': avEntry
                    }
                    #st.write(appdetails)
                    column = list(appdetails.keys())
                    new_data_frame = pd.DataFrame([appdetails],columns = column)
                    df = df.append(new_data_frame,ignore_index=True,sort = False) 
                    df.to_csv("/Users/ruudrapatel/Desktop/Study/PythonProgs/PyInterns/Appdata.csv",index=False)
                    curr.execute("""INSERT INTO CleanedData ('App','Category','Rating','Reviews','Size','Installs','Type','Price','ContentRating','Genres','LastUpdated','CurrentVer','AndroidVer') values (?,?,?,?,?,?,?,?,?,?,?,?,?)""",(appEntry,catEntry,ratEntry,reEntry,sizeEntry,insEntry,typeEntry,priceEntry,conEntry,gEntry,luEntry,cvEntry,avEntry))
                else:
                    st.warning("Missing Values")
    with col2:
        st.header("Enter Reviews Data")
        with st.form(key="Form2"):
            appE = st.text_input("App Name")
            trEntry = st.text_area("Translated Review")
            sEntry = st.selectbox("Sentiment",list3)
            spEntry = st.number_input("Sentiment Polarity",min_value=-1.0,max_value=1.0)
            ssEntry = st.number_input("Sentiment Subjectivity",min_value=0.0,max_value=1.0)
            if st.form_submit_button("Submit App Reviews"):
                if appE != "":
                    st.success("Reviews Added for "+appE)
                    #st.write(appE,trEntry,sEntry,spEntry,ssEntry)
                    review_details ={
                    'App':appE,
                    'Translated_Review':trEntry,
                    'Sentiment':sEntry,
                    'Sentiment_ Polarity': spEntry,
                    'Sentiment_Subjectivity':ssEntry
                    }
                    column = list(review_details.keys())
                    new_data_frame = pd.DataFrame([review_details],columns = column)
                    review_df = review_df.append(new_data_frame,ignore_index = True , sort = False) 
                    review_df.to_csv("/Users/ruudrapatel/Desktop/Study/PythonProgs/PyInterns/Reviewdata.csv",index=False)
                    curr.execute("""INSERT INTO ReviewsTable('App','Translated_Review','Sentiment','Sentiment_Polarity','Sentiment_Subjectivity') values(?,?,?,?,?)""",(appE,trEntry,sEntry,spEntry,ssEntry))
                else:
                    st.warning("Missing Values")



def ploting_chart_two_column(col,title,xtitle,plot_type=go.Scatter):
    v1=data_free[col].value_counts().reset_index()
    v1=v1.rename(columns={col:'count','index':col})

    v1['percent']=v1['count'].apply(lambda x : 100*x/sum(v1['count']))
    v1=v1.sort_values(col)
    
    v2=data_paid[col].value_counts().reset_index()
    v2=v2.rename(columns={col:'count','index':col})
    v2['percent']=v2['count'].apply(lambda x : 100*x/sum(v2['count']))
    v2=v2.sort_values(col)
    
    trace1 =  plot_type(x=v1[col], y=v1["count"], name="Free", marker=dict(color="#a678de"))
    trace2 = plot_type(x=v2[col], y=v2["count"], name="Paid", marker=dict(color="#6ad49b"))
    y = [trace1,trace2]
    layout={'title':title,'xaxis':{'title':xtitle}}
    fig = go.Figure(data=y, layout=layout)
    return st.plotly_chart(fig)


def feature19():
    #Part 1
    column = 'Type'
    grouped = df[column].value_counts().reset_index()
    grouped = grouped.rename(columns={column:'count','index':column})
    trace = go.Pie(labels=grouped[column],values=grouped['count'],pull=[0.05,0])
    layout = {'title':'Percent of Free/Paid Apps'}
    fig1 = go.Figure(data=[trace],layout=layout)
    st.plotly_chart(fig1)
    
    #Part 2
    ploting_chart_two_column('Rating','Rating of the free and Paid Apps','Rating',go.Scatter)


def feature20():
    ploting_chart_two_column('Android Ver','Android Version','All Versions')
    ploting_chart_two_column('Content Rating','Content Rating of Apps FREE VS PAID','Rating',go.Scatter)
    
    
def feature21():
    st.markdown("<h1 style='text-align: center; color: green;'>Thank You</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: white;'>By Bhargav Mishal, Ruudra Patel and Jash Shah</p>",unsafe_allow_html=True)
    st.balloons()



st.markdown("<h1 style='text-align: center; color: white;,font:Sans serif'>Exploratory Data Analysis on Play Store</h1>", unsafe_allow_html=True)

with st.sidebar:
	st.header("FEATURES")
	#qno_dropbox = st.selectbox('Choose a feature :', [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19], key='qno')
	qno = st.number_input('Select a feature :',value=1,min_value=1,max_value=21)


questions = ['1) What is the percentage download in each category on the playstore',
            '''2) How many apps have managed to get the following number of downloads
            a) Between 10,000 and 50,000
            b) Between 50,000 and 150000
            c) Between 150000 and 500000
            d) Between 500000 and 5000000
            e) More than 5000000''',
            '3) Which category of apps have managed to get the most,least and an average of 2,50,000 downloads atleast',
            '4) Which category of apps have managed to get the highest maximum average ratings from the users.Display the result using suaitable visualization tool(s) and also update the data into the database',
            '''5) What is the number of installs for the following app sizes.
            a) Size between 10 and 20 mb
            b) Size between 20 and 30 mb
            c) More than 30 mb''',
            '6) For the years 2016,2017,2018 what are the category of apps that have got the most and the least downloads',
            '7) All those apps , whose android version is not an issue and can work with varying devices ,what is the percentage increase or decrease in the downloads',
            '8) Amongst sports, entertainment,social media,news,events,travel and games,which is the category of app that is most likely to be downloaded in the coming years, kindly make a prediction and back it with suitable findings.Also update the number of downloads that these categories have received into a database',
            '9) All those apps who have managed to get over 1,00,000 downloads, have they managed to get an average rating of 4.1 and above? An we conclude something in co-relation to the number of downloads and the ratings received',
            '10) Across all the years ,which month has seen the maximum downloads fr each of the category. What is the ratio of downloads for the app that qualifies as teen versus mature17+',
            '11) Which quarter of which year has generated the highest number of install for each app used in the study?',
            '12) Which of all the apps given have managed to generate the most positive and negative sentiments.Also figure out the app which has generated approximately the same ratio for positive and negative sentiments',
            '13) Study and find out the relation between the Sentiment-polarity and sentimentsubjectivity of all the apps. What is the sentiment subjectivity for a sentiment polarity of 0.4',
            '14) Generate an interface where the client can see the reviews categorized as positive.negative and neutral ,once they have selected the app from a list of apps available for the study',
            '15) Is it advisable to launch an app like ’10 Best foods for you’? Do the users like these apps?',
            '16) Which month(s) of the year , is the best indicator to the avarage downloads that an app will generate over the entire year?',
            '17) Does the size of the App influence the number of installs that it gets ? if,yes the trend is positive or negative with the increase in the app size',
            '18) Provide an interface to add new data to both the datasets provided.The data needs to be added to the excel sheets.',
            '19) Free vs Paid',
            '20) Android Version and Content Rating for Free/Paid apps',
            ''
]

st.write(questions[qno-1])

f_results = eval('feature'+str(qno)+'()')

curr.close()
con.close()