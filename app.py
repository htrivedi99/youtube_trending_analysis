import dash
import dash_core_components as dcc
import dash_html_components as html
import plotly
import plotly.graph_objs as go
import plotly.figure_factory as ff
import pandas as pd
import numpy as np
import json
import nltk
import re
import random


app = dash.Dash(__name__)
server = app.server


# CA_data = pd.read_csv('data/CAvideos.csv') #Canada
FR_data = pd.read_csv('data/FRvideos.csv') #France
# GB_data = pd.read_csv('data/GBvideos.csv') #UK
# DE_data = pd.read_csv('data/DEvideos.csv') #Germany
US_data = pd.read_csv('data/USvideos.csv') #US


def add_category_name():
    # with open('data/CA_category_id.json') as f:
    #     categories = json.load(f)["items"]
    # category_dict = {}
    # for category in categories:
    #     category_dict[int(category["id"])] = category["snippet"]["title"]
    # CA_data['category_name'] = CA_data["category_id"].map(category_dict)

    with open('data/US_category_id.json') as f:
        categories = json.load(f)["items"]
    category_dict = {}
    for category in categories:
        category_dict[int(category["id"])] = category["snippet"]["title"]
    US_data['category_name'] = US_data["category_id"].map(category_dict)

    with open('data/FR_category_id.json') as f:
        categories = json.load(f)["items"]
    category_dict = {}
    for category in categories:
        category_dict[int(category["id"])] = category["snippet"]["title"]
    FR_data['category_name'] = FR_data["category_id"].map(category_dict)
    #
    # with open('data/DE_category_id.json') as f:
    #     categories = json.load(f)["items"]
    # category_dict = {}
    # for category in categories:
    #     category_dict[int(category["id"])] = category["snippet"]["title"]
    # DE_data['category_name'] = DE_data["category_id"].map(category_dict)
    #
    # with open('data/GB_category_id.json') as f:
    #     categories = json.load(f)["items"]
    # category_dict = {}
    # for category in categories:
    #     category_dict[int(category["id"])] = category["snippet"]["title"]
    # GB_data['category_name'] = GB_data["category_id"].map(category_dict)

add_category_name()

US_data['region'] = 'US'
FR_data['region'] = 'FR'
# GB_data['region'] = 'GB'
# DE_data['region'] = 'DE'
# CA_data['region'] = 'CA'

#df = pd.concat([CA_data, FR_data, GB_data, DE_data, US_data])
df = pd.concat([US_data, FR_data])

df['year_published'] = df['publish_time'].astype(str).str[:4]
df['year_published'] = df['year_published'].astype('int')

df['year_trending'] = '20' + df['trending_date'].astype(str).str[:2]
df['year_trending'] = df['year_trending'].astype(int)

# print(df[df['region'] == 'US' & df['category_name'] == ''])
#print(df.info())
#print(df.head())

category_names = US_data['category_name'].unique()
#print(category_names)

def clean_text(sentence):
    result = []
    words = []
    exp = '\w+'
    tokenized = re.findall(exp, sentence)
    for word in tokenized:
        words.append(word.lower())
    sw = nltk.corpus.stopwords.words('english')
    for word in words:
        if word not in sw:
            result.append(word)
    return result



app.layout = html.Div([

    html.Div([
        html.Div([
            html.H1('Youtube Trending Analysis')
        ],style={'font-family':'Courier New, monospace', 'color':'#FF4343', 'font-size':16}),

        html.Div([
            html.Div([
                #html.Label('Country'),
                dcc.Dropdown(
                id='country-column',
                options=[
                {'label':'United States', 'value':'US'},
                # {'label':'Canada', 'value':'CA'},
                # {'label':'United Kingdom', 'value':'GB'},
                {'label':'France', 'value':'FR'},
                # {'label':'Germany', 'value':'DE'}
                ],
                value=['US'],
                multi=True
            ),
            ], style={'width': '100%','display': 'inline-block', 'float':'left'}),
            html.Div([
                dcc.Graph(id='views-like-scatter')
            ], style={'width': '100%','display': 'inline-block', 'padding': '0 20', 'backgroundColor': 'blue', 'float':'left'}),
            #'padding': '0 20'

            html.Div(dcc.Slider(
            id='year-slider',
            min=df['year_published'].min(),
            max=df['year_published'].max(),
            value=df['year_published'].max(),
            marks={str(year): str(year) for year in df['year_published'].unique()}
            ),style={'width': '90%', 'float':'left', 'padding': '0px 20px 20px 20px'}),
            #'padding': '0px 20px 20px 20px'

        ], style={'float' : 'left',
                  'width': '65%',
                 'display' : 'inline-block',
                 'boxSizing' : 'border-box'}),

        html.Div([

            html.Div([
                #html.Label('Youtube Category'),
                dcc.Dropdown(
                id='category-column',
                options=[
                {'label':'Autos & Vechicles', 'value':'Autos & Vehicles'},
                {'label':'Comedy', 'value':'Comedy'},
                {'label':'Education', 'value':'Education'},
                {'label':'Entertainment', 'value':'Entertainment'},
                {'label':'Film & Amimation', 'value':'Film & Animation'},
                {'label':'Gaming', 'value':'Gaming'},
                {'label':'Howto & style', 'value':'Howto & Style'},
                {'label':'Music', 'value':'Music'},
                {'label':'News & Politics', 'value':'News & Politics'},
                {'label':'Nonprofits & Activism', 'value':'Nonprofits & Activism'},
                {'label':'People & Blogs', 'value':'People & Blogs'},
                {'label':'Pets & Animals', 'value':'Pets & Animals'},
                {'label':'Science & Technology', 'value':'Science & Technology'},
                {'label':'Shows', 'value':'Shows'},
                {'label':'Sports', 'value':'Sports'},
                {'label':'Travel & Events', 'value':'Travel & Events'},
                ],
                value='Gaming',
                multi=False
            ),
            ], style={'width':'100%', 'float':'right', 'display': 'inline-block'}),


            html.Div([
                dcc.Graph(
                id='comments_bar',
    )], style={'width':'45%', 'float':'left', 'display': 'inline-block', 'padding':'8px 8px 8px 8px'}),
            html.Div([
                dcc.Graph(
                id='dislikes_bar',
    )], style={'width':'45%', 'float':'right', 'display': 'inline-block','padding':'8px 16px 8px 8px'}),

        html.Div([
            dcc.Graph(
            id='pop_channel_bar',
)], style={'width':'45%', 'float':'left', 'display': 'inline-block', 'padding':'0px 8px 8px 8px'}),


        ], style={'width':'35%', 'float':'right', 'display': 'inline-block', 'boxSizing' : 'border-box'}),
        html.Div([
            html.Div([
                dcc.Graph(
                    id='wordcloud',
                )
            ],style={'width':'100%', 'float':'left', 'display': 'inline-block'}),
            ], style={'float' : 'left',
                      'width': '65.5%',
                     'display' : 'inline-block',
                     'boxSizing' : 'border-box',
                     'padding':'8px 8px 8px 8px'}),

        html.Div([
            html.Div([
                #html.Label('Youtube Category'),
                dcc.Dropdown(
                id='cat-col',
                options=[
                {'label':'Autos & Vechicles', 'value':'Autos & Vehicles'},
                {'label':'Comedy', 'value':'Comedy'},
                {'label':'Education', 'value':'Education'},
                {'label':'Entertainment', 'value':'Entertainment'},
                {'label':'Film & Amimation', 'value':'Film & Animation'},
                {'label':'Gaming', 'value':'Gaming'},
                {'label':'Howto & style', 'value':'Howto & Style'},
                {'label':'Music', 'value':'Music'},
                {'label':'News & Politics', 'value':'News & Politics'},
                {'label':'Nonprofits & Activism', 'value':'Nonprofits & Activism'},
                {'label':'People & Blogs', 'value':'People & Blogs'},
                {'label':'Pets & Animals', 'value':'Pets & Animals'},
                {'label':'Science & Technology', 'value':'Science & Technology'},
                {'label':'Shows', 'value':'Shows'},
                {'label':'Sports', 'value':'Sports'},
                {'label':'Travel & Events', 'value':'Travel & Events'},
                ],
                value='Pets & Animals',
                multi=False
            ),
            ], style={'width':'50%', 'float':'left', 'display': 'inline-block'}),

            html.Div([
                #html.Label('Youtube Category'),
                dcc.Dropdown(
                id='vars-col',
                options=[
                {'label':'Views', 'value':'views'},
                {'label':'Likes', 'value':'likes'},
                {'label':'Dislikes', 'value':'dislikes'},
                {'label':'# Comments', 'value':'comment_count'},
                ],
                value='views',
                multi=False
            ),
            ], style={'width':'50%', 'float':'right', 'display': 'inline-block'}),

            html.Div([
                dcc.Graph(
                    id='histogram',
                )
            ],style={'width':'100%', 'float':'right', 'display': 'inline-block'}),
            ], style={'float' : 'right',
                      'width': '34.5%',
                     'display' : 'inline-block',
                     'boxSizing' : 'border-box'}),
        ])
])

@app.callback(
   dash.dependencies.Output('views-like-scatter', 'figure'),
   [dash.dependencies.Input('year-slider', 'value'),
   dash.dependencies.Input('country-column', 'value')])

   #regions = ['US','FR','GB','DE','CA']

def update_scatterplot_graph(year, region):
    df_year = df[df['year_published'] == year]
    dff = df_year[df_year['region'].isin(region)]
    sampled_df = pd.DataFrame()

    size = 100
    replace = True
    fn = lambda obj: obj.loc[np.random.choice(obj.index, size, replace),:]
    sampled_df = dff.groupby('category_name', as_index=False).apply(fn)

    traces = []
    for i in sampled_df['category_name'].unique():
        df_by_category = sampled_df[sampled_df['category_name'] == i]

        traces.append(go.Scatter(
            x=df_by_category['views'],
            y=df_by_category['likes'],
            text=df_by_category['views'],
            mode='markers',
            opacity=0.6,
            marker={
                'size': 12,
                'line': {'width': 0.5, 'color': 'white'}
            },
            name=i
        ))
    return {
        'data': traces,
        'layout': go.Layout(
            xaxis= {'title': 'Number of Views', 'range': [0,10000000]},
            yaxis= {'title': 'Number of Likes', 'range': [0,1000000]},
            margin={'l': 50, 'b': 40, 't': 10, 'r': 10},
            legend={'x': 0, 'y': 1},
            paper_bgcolor='rgb(240,248,255)',
            plot_bgcolor='rgb(240,248,255)',
            hovermode='closest'
        )
    }



@app.callback(
   dash.dependencies.Output('comments_bar', 'figure'),
   [dash.dependencies.Input('category-column', 'value'),
   dash.dependencies.Input('country-column', 'value')])

def update_comments_bar_plot(category, region):
    dff = df[df['region'].isin(region)]
    dff['month'] = dff['trending_date'].astype(str).str[6:]
    dff['month'] = dff['month'].astype(int)
    category_filter = dff[dff['category_name'] == category]
    num_views = []
    month_numbers = dff['month'].unique()
    for i in month_numbers:
        filter1 = category_filter[category_filter['month'] == i]
        sum = filter1['comment_count'].sum()
        num_views.append(sum)

    traces = []

    traces.append(go.Bar(
        x=month_numbers,
        y=num_views,
        marker=dict(color='#A5FAFF'),
        opacity=0.75))
    return {
        'data':traces,
        'layout': go.Layout(
            title='Comments per month',
            #xaxis= {'title': 'Month'},
            #yaxis= {'title': 'Number of Comments'},
            height=235,
            width=235,
            margin={'l': 40,'r': 5,'t': 35,'b': 20},
            bargap=0.2,
            paper_bgcolor='rgb(255,201,92)',
            plot_bgcolor='rgb(255,201,92)',
        )
    }

@app.callback(
   dash.dependencies.Output('dislikes_bar', 'figure'),
   [dash.dependencies.Input('category-column', 'value'),
   dash.dependencies.Input('country-column', 'value')])

def update_dislikes_bar(category, region):
    dff = df[df['region'].isin(region)]
    dff['month'] = dff['trending_date'].astype(str).str[6:]
    dff['month'] = dff['month'].astype(int)
    category_filter = dff[dff['category_name'] == category]
    num_dislikes = []
    month_numbers = dff['month'].unique()
    for i in month_numbers:
        filter1 = category_filter[category_filter['month'] == i]
        sum = filter1['dislikes'].sum()
        num_dislikes.append(sum)

    traces = []

    traces.append(go.Bar(
        x=month_numbers,
        y=num_dislikes,
        marker=dict(color='#FFFC7F'),
        opacity=0.75))
    return {
        'data':traces,
        'layout': go.Layout(
            title='Dislikes per month',
            height=235,
            width=235,
            margin={'l': 40,'r': 5,'t': 35,'b': 20},
            bargap=0.2,
            paper_bgcolor='rgb(254,215,255)',
            plot_bgcolor='rgb(254,215,255)',
        )
    }

@app.callback(
   dash.dependencies.Output('pop_channel_bar', 'figure'),
   [dash.dependencies.Input('category-column', 'value'),
   dash.dependencies.Input('country-column', 'value')])

def update_pop_channel_bar(category, region):
    dff = df[df['region'].isin(region)]
    dff['month'] = dff['trending_date'].astype(str).str[6:]
    dff['month'] = dff['month'].astype(int)
    category_filter = dff[dff['category_name'] == category]
    cdf = category_filter.groupby("channel_title").size().reset_index(name="video_count") \
        .sort_values("video_count", ascending=False).head(10)
    channel_titles = []
    video_counts = []
    for i in cdf:
        channel_titles.append(cdf['channel_title'])
        video_counts.append(cdf['video_count'])
    np_titles = np.array(channel_titles)
    np_vcounts = np.array(video_counts)
    raveled_titles = np_titles.ravel()
    raveled_vcounts = np_vcounts.ravel()
    traces = []
    traces.append(go.Bar(
        x=raveled_vcounts,
        y=raveled_titles,
        orientation='h',
        marker=dict(color='#98FB98'),
        opacity=0.75))
    return {
        'data':traces,
        'layout': go.Layout(
            title='Most Videos by Category',
            height=235,
            width=490,
            margin={'l': 130,'r': 5,'t': 35,'b': 20},
            bargap=0.2,
            paper_bgcolor='rgb(240,140,140)',
            plot_bgcolor='rgb(240,140,140)',
        )
    }

@app.callback(
   dash.dependencies.Output('wordcloud', 'figure'),
   [dash.dependencies.Input('category-column', 'value'),
   dash.dependencies.Input('country-column', 'value')])



def update_word_cloud(category, region):
    dff = df[df['region'].isin(region)]
    category_filter = dff[dff['category_name'] == category]
    titles_col = category_filter['title']
    words = []
    for i in titles_col:
        cleaned_words = clean_text(i)
        words.append(cleaned_words)
    flattened_words = np.hstack(words)
    words_length = len(flattened_words)
    colors = [plotly.colors.DEFAULT_PLOTLY_COLORS[random.randrange(1, 10)] for i in range(words_length)]
    freq_dist = nltk.FreqDist(flattened_words)
    keys = freq_dist.keys()
    h_keys = np.hstack(keys)
    values = freq_dist.values()
    h_values = np.hstack(values)
    lower, upper = 15, 45
    h_values = [((x - min(h_values)) / (max(h_values) - min(h_values))) * (upper - lower) + lower for x in h_values]
    traces = []
    traces.append(go.Scatter(
        x=[random.random() for i in range(100)],
        y=random.choices(range(words_length), k=words_length),
        #random.choices(range(30), k=30)
        mode='text',
        text=h_keys,
        opacity=0.7,
        textfont={'size':h_values, 'color':colors}
    ))
    return{
        'data':traces,
        'layout': go.Layout(
            xaxis= {'showgrid':False, 'showticklabels':False, 'zeroline':False},
            yaxis= {'showgrid':False, 'showticklabels':False, 'zeroline':False},
            height=480,
            #width=825,
            margin={'l': 40,'r': 5,'t': 30,'b': 20},
            paper_bgcolor='rgb(47,79,79)',
            plot_bgcolor='rgb(47,79,79)',
            #hovermode='closest'
        )
    }


@app.callback(
   dash.dependencies.Output('histogram', 'figure'),
   [dash.dependencies.Input('cat-col', 'value'),
   dash.dependencies.Input('vars-col', 'value')])

def update_histogram(category, col_name):
    dff = df[df['category_name'] == category]
    filtered = dff[col_name]
    traces = []
    traces.append(go.Histogram(
    x=filtered,
    marker=dict(
        color='#35B8C1'
    )

    ))

    return{
        'data':traces,
        'layout': go.Layout(
            title='Category Histogram',
            margin={'l': 40,'r': 5,'t': 30,'b': 20},
            paper_bgcolor='rgb(242,255,174)',
            plot_bgcolor='rgb(242,255,174)',
            xaxis={'range': [0,7000000]},
            #hovermode='closest'
        )
    }


if __name__ == '__main__':
    app.run_server(debug=True)
