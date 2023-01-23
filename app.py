import pathlib
import json
from datetime import datetime
import dash
import dash_table
import matplotlib.colors as mcolors
import dash_bootstrap_components as dbc
import dash_core_components as dcc
import dash_html_components as html
import plotly.graph_objs as go
import plotly.express as px
import pandas as pd
import numpy as np
from precomputing import add_stopwords
from dash.dependencies import Output, Input, State
from dateutil import relativedelta
from wordcloud import WordCloud, STOPWORDS

filmweb_df = pd.read_csv('data/df_filmweb.csv')#.sample(100)
imdb_df = pd.read_csv('data/df_imdb.csv').sample(10000)
filmweb_df['description_length'] = filmweb_df['description'].str.len()
imdb_df['description_length'] = imdb_df['description'].str.len()

PLOTLY_LOGO = "https://images.plot.ly/logo/new-branding/plotly-logomark.png"

ADDITIONAL_STOPWORDS = [
    "XXXX", "XX", "xx", "xxxx", "n't", "Trans Union", "BOA", "Citi", "account", "się", "na",
    "jego", "nbsp", "jest", "po", "przez", "jej", "który", "nie", "aby", "od", "go", "jednak", "Gdy", "ich", "przed",
    "że", "za", "która", "ma", "dla", "kiedy", "oraz", "są", "do", "tego", "także", "tak", "jest", "był", "była", "było", "ją", "jako",
    "ze"
]
for stopword in ADDITIONAL_STOPWORDS:
    STOPWORDS.add(stopword)


def sample_data(dataframe, float_percent):
    """
    Returns a subset of the provided dataframe.
    The sampling is evenly distributed and reproducible
    """
    print("making a local_df data sample with float_percent: %s" % (float_percent))
    return dataframe.sample(frac=float_percent, random_state=1)


def get_complaint_count_by_company(dataframe):
    """ Helper function to get complaint counts for unique banks """
    company_counts = dataframe["Company"].value_counts()
    # we filter out all banks with less than 11 complaints for now
    company_counts = company_counts[company_counts > 10]
    values = company_counts.keys().tolist()
    counts = company_counts.tolist()
    return values, counts


def make_marks_time_slider(mini, maxi):
    """
    A helper function to generate a dictionary that should look something like:
    {1420066800: '2015', 1427839200: 'Q2', 1435701600: 'Q3', 1443650400: 'Q4',
    1451602800: '2016', 1459461600: 'Q2', 1467324000: 'Q3', 1475272800: 'Q4',
     1483225200: '2017', 1490997600: 'Q2', 1498860000: 'Q3', 1506808800: 'Q4'}
    """
    step = relativedelta.relativedelta(months=+12)
    start = datetime(year=mini.year, month=1, day=1)
    end = datetime(year=maxi.year, month=maxi.month, day=30)
    ret = {}

    current = start
    while current <= end:
        current_str = int(current.timestamp())
        # print(current.year)
        if current.year % 8 == 0:
            ret[current_str] = {
                "label": str(current.year),
                "style": {"font-weight": "bold"},
            }
        else:
            ret[current_str] = {
                "label": "",
                "style": {"font-weight": "bold"},
            }
        current += step
    return ret


def time_slider_to_date(time_values):
    min_date = datetime.fromtimestamp(time_values[0]).strftime("%Y")
    max_date = datetime.fromtimestamp(time_values[1]).strftime("%Y")
    print("Converted time_values: ")
    print("\tmin_date:", time_values[0], "to: ", min_date)
    print("\tmax_date:", time_values[1], "to: ", max_date)
    return [min_date, max_date]


def make_options_bank_drop(values):
    ret = []
    for value in values:
        ret.append({"label": value, "value": value})
    return ret


def populate_lda_scatter(selected_bank, plot_option):
    traces = []
    if selected_bank == 'Filmweb':
        local_df = filmweb_df[filmweb_df['year'].notna()]
    elif selected_bank == 'IMDB':
        local_df = imdb_df[imdb_df['year'].notna()]
    print('SELECTE$D BANC', selected_bank)
    for genre in local_df['genre'].unique():
        filmweb_df_genre = local_df[local_df['genre'] == genre]
        titles = filmweb_df_genre['title'].tolist()

        genres = []
        for _ in range(len(titles)):
            genres.append(genre)

        description_lengths = filmweb_df_genre['description_length'] + np.random.uniform(-0.2, 0.2, len(filmweb_df_genre))

        if plot_option == 'mean_description':
            years = filmweb_df_genre['year'] + np.random.uniform(-0.2, 0.2, len(filmweb_df_genre))
            description_lengths = filmweb_df_genre['description_length'] + np.random.uniform(-0.2, 0.2, len(filmweb_df_genre))
            x = years
            y = description_lengths
            hovers = titles

        elif plot_option == 'count_movies':
            another_local_df = filmweb_df_genre.groupby('year')['title'].count().reset_index()

            years = another_local_df['year'].tolist()
            title = another_local_df['title'].tolist()
            x = years
            y = title
            hovers = genres

        trace = go.Scatter(
            name=genre,
            x=x,
            y=y,
            mode="markers",
            hovertext=hovers,
            marker=dict(
                size=6,
                colorscale="Viridis",
                showscale=False,
            ),
        )
        traces.append(trace)

    layout = go.Layout({"title": "Release year, genre and description length"})

    return {"data": traces, "layout": layout}


def plotly_wordcloud(data_frame):
    complaints_text = list(data_frame["description"].dropna().values)

    if len(complaints_text) < 1:
        return {}, {}, {}

    # join all documents in corpus
    text = " ".join(list(complaints_text))

    word_cloud = WordCloud(stopwords=set(STOPWORDS), max_words=100, max_font_size=90)
    word_cloud.generate(text)

    word_list = []
    freq_list = []
    fontsize_list = []
    position_list = []
    orientation_list = []
    color_list = []

    for (word, freq), fontsize, position, orientation, color in word_cloud.layout_:
        word_list.append(word)
        freq_list.append(freq)
        fontsize_list.append(fontsize)
        position_list.append(position)
        orientation_list.append(orientation)
        color_list.append(color)

    # get the positions
    x_arr = []
    y_arr = []
    for i in position_list:
        x_arr.append(i[0])
        y_arr.append(i[1])

    # get the relative occurence frequencies
    new_freq_list = []
    for i in freq_list:
        new_freq_list.append(i * 80)

    trace = go.Scatter(
        x=x_arr,
        y=y_arr,
        textfont=dict(size=new_freq_list, color=color_list),
        hoverinfo="text",
        textposition="top center",
        hovertext=["{0} - {1}".format(w, f) for w, f in zip(word_list, freq_list)],
        mode="text",
        text=word_list,
    )

    layout = go.Layout(
        {
            "xaxis": {
                "showgrid": False,
                "showticklabels": False,
                "zeroline": False,
                "automargin": True,
                "range": [-100, 250],
            },
            "yaxis": {
                "showgrid": False,
                "showticklabels": False,
                "zeroline": False,
                "automargin": True,
                "range": [-100, 450],
            },
            "margin": dict(t=20, b=20, l=10, r=10, pad=4),
            "hovermode": "closest",
        }
    )

    wordcloud_figure_data = {"data": [trace], "layout": layout}
    word_list_top = word_list[:25]
    word_list_top.reverse()
    freq_list_top = freq_list[:25]
    freq_list_top.reverse()

    frequency_figure_data = {
        "data": [
            {
                "y": word_list_top,
                "x": freq_list_top,
                "type": "bar",
                "name": "",
                "orientation": "h",
            }
        ],
        "layout": {"height": "550", "margin": dict(t=20, b=20, l=100, r=20, pad=4)},
    }
    treemap_trace = go.Treemap(
        labels=word_list_top, parents=[""] * len(word_list_top), values=freq_list_top
    )
    treemap_layout = go.Layout({"margin": dict(t=10, b=10, l=5, r=5, pad=4)})
    treemap_figure = {"data": [treemap_trace], "layout": treemap_layout}
    return wordcloud_figure_data, frequency_figure_data, treemap_figure


NAVBAR = dbc.Navbar(
    children=[
        html.A(
            # Use row and col to control vertical alignment of logo / brand
            dbc.Row(
                [
                    dbc.Col(html.Img(src=PLOTLY_LOGO, height="30px")),
                    dbc.Col(
                        dbc.NavbarBrand("Filmweb and IMDB Comparison", className="ml-2")
                    ),
                ],
                align="center",
                no_gutters=True,
            ),
            href="https://plot.ly",
        )
    ],
    color="dark",
    dark=True,
    sticky="top",
)

LEFT_COLUMN = dbc.Jumbotron(
    [
        html.H4(children="Select dataset and time frame", className="display-5"),
        html.Hr(className="my-2"),
        html.Label("Select dataset", style={"marginTop": 50}, className="lead"),
        html.P(
            "(You can use the dropdown or click the barchart on the right)",
            style={"fontSize": 10, "font-weight": "lighter"},
        ),
        dcc.Dropdown(
            id="bank-drop", clearable=False, style={"marginBottom": 50, "font-size": 12}
        ),
        html.Label("Select time frame", className="lead"),
        html.Div(dcc.RangeSlider(id="time-window-slider"), style={"marginBottom": 50, 'width': '400px', 'marginLeft':0}),
    ]
)

LEFT_COLUMN2 = dbc.Jumbotron(
    [
        html.H4(children="Select dataset and genre", className="display-5"),
        html.Hr(className="my-2"),
        html.Label("Select dataset", style={"marginTop": 50}, className="lead"),
        html.P(
            "(You can use the dropdown or click the barchart on the right)",
            style={"fontSize": 10, "font-weight": "lighter"},
        ),
        dcc.Dropdown(
            id="bank-drop22", clearable=False, style={"marginBottom": 50, "font-size": 12}, value="imdb"
        ),
        html.Label("Select genre", style={"marginTop": 50}, className="lead"),
        html.P(
            "(You can use the dropdown or click the barchart on the right)",
            style={"fontSize": 10, "font-weight": "lighter"},
        ),
        dcc.Dropdown(
            id="bank-drop23", clearable=False, style={"marginBottom": 50, "font-size": 12}, value='comedy'
        ),
        html.Div(dcc.RangeSlider(id="time-window-slider22"), style={"display":"none", "marginBottom": 50, 'width': '400px', 'marginLeft':0}),
    ]
)

LDA_PLOT = dcc.Loading(
    id="loading-lda-plot", children=[dcc.Graph(id="tsne-lda")], type="default"
)
LDA_TABLE = html.Div(
    id="lda-table-block",
    children=[
        dcc.Loading(
            id="loading-lda-table",
            children=[
                dash_table.DataTable(
                    id="lda-table",
                    style_cell_conditional=[
                        {
                            "if": {"column_id": "Text"},
                            "textAlign": "left",
                            "whiteSpace": "normal",
                            "height": "auto",
                            "min-width": "50%",
                        }
                    ],
                    style_data_conditional=[
                        {
                            "if": {"row_index": "odd"},
                            "backgroundColor": "rgb(243, 246, 251)",
                        }
                    ],
                    style_cell={
                        "padding": "16px",
                        "whiteSpace": "normal",
                        "height": "auto",
                        "max-width": "0",
                    },
                    style_header={"backgroundColor": "white", "fontWeight": "bold"},
                    style_data={"whiteSpace": "normal", "height": "auto"},
                    filter_action="native",
                    page_action="native",
                    page_current=0,
                    page_size=5,
                    columns=[],
                    data=[],
                )
            ],
            type="default",
        )
    ],
    style={"display": "none"},
)

LDA_PLOTS = [
    dbc.CardHeader(html.H5("All movies scatter plot")),
    dbc.Alert(
        "Not enough data to render LDA plots, please adjust the filters",
        id="no-data-alert-lda",
        color="warning",
        style={"display": "none"},
    ),
    dbc.CardBody(
        [
            dcc.Dropdown(
                id="bank-drop2", clearable=False, style={"marginBottom": 50, "font-size": 12}, value="Filmweb",
            ),
            dcc.Dropdown(
                id="bank-drop3",
                options=[{"label": "Description length", "value": "mean_description"}, {"label": "Count movies", "value": "count_movies"}],
                value="count_movies",
            ),
            LDA_PLOT,
            html.Hr(),
            LDA_TABLE,
        ]
    ),
]
WORDCLOUD_PLOTS = [
    dbc.CardHeader(html.H5("Most frequently used words in descriptions")),
    dbc.Alert(
        "Not enough data to render these plots, please adjust the filters",
        id="no-data-alert",
        color="warning",
        style={"display": "none"},
    ),
    dbc.CardBody(
        [
            dbc.Row(
                [
                    dbc.Col(
                        dcc.Loading(
                            id="loading-frequencies",
                            children=[dcc.Graph(id="frequency_figure")],
                            type="default",
                        )
                    ),
                    dbc.Col(
                        [
                            dcc.Tabs(
                                id="tabs",
                                children=[
                                    dcc.Tab(
                                        label="Treemap",
                                        children=[
                                            dcc.Loading(
                                                id="loading-treemap",
                                                children=[dcc.Graph(id="bank-treemap")],
                                                type="default",
                                            )
                                        ],
                                    ),
                                    dcc.Tab(
                                        label="Wordcloud",
                                        children=[
                                            dcc.Loading(
                                                id="loading-wordcloud",
                                                children=[
                                                    dcc.Graph(id="bank-wordcloud")
                                                ],
                                                type="default",
                                            )
                                        ],
                                    ),
                                ],
                            )
                        ],
                        md=8,
                    ),
                ]
            ),
            dbc.Row(
                [
                    dbc.Col(html.P("Choose dataset and genre:"), md=12),
                    dbc.Col(
                        [
                            dcc.Dropdown(
                                id="bigrams-comp_3",
                                options=[
                                    {"label": "IMDB", "value": "IMDB"}, {"label": "Filmweb", "value": "Filmweb"}
                                ],
                                value="IMDB",
                            )
                        ],
                        md=6,
                    ),
                    dbc.Col(
                        [
                            dcc.Dropdown(
                                id="bigrams-comp_4",
                                options=[
                                    {"label": i, "value": i}
                                    for i in filmweb_df.genre.unique()
                                ] + [{"label": "ALL", "value": "ALL"}],
                                value="comedy",
                            )
                        ],
                        md=6,
                    ),
                ]
            ),
        ]
    ),
]

TOP_BANKS_PLOT = [
    dbc.CardHeader(html.H5("Top 10 most popular genres in time")),
    dbc.CardBody(
        [
            dcc.Loading(
                id="loading-banks-hist",
                children=[
                    dbc.Alert(
                        "Not enough data to render this plot, please adjust the filters",
                        id="no-data-alert-bank",
                        color="warning",
                        style={"display": "none"},
                    ),
                    dcc.Graph(id="bank-sample"),
                ],
                type="default",
            )
        ],
        style={"marginTop": 0, "marginBottom": 0},
    ),
]

TOP_BANKS_PLOT2 = [
    dbc.CardHeader(html.H5("Top 10 TF-IDF most significant words")),
    dbc.CardBody(
        [
            dcc.Loading(
                id="loading-banks-hist22",
                children=[
                    dbc.Alert(
                        "Not enough data to render this plot, please adjust the filters",
                        id="no-data-alert-bank22",
                        color="warning",
                        style={"display": "none"},
                    ),
                    dcc.Graph(id="bank-sample22"),
                ],
                type="default",
            )
        ],
        style={"marginTop": 0, "marginBottom": 0},
    ),
]

TOP_BIGRAM_COMPS = [
    dbc.CardHeader(html.H5("Comparison of two datasets")),
    dbc.CardBody(
        [
            dcc.Loading(
                id="loading-bigrams-comps",
                children=[
                    dbc.Alert(
                        "Something's gone wrong! Give us a moment, but try loading this page again if problem persists.",
                        id="no-data-alert-bigrams_comp",
                        color="warning",
                        style={"display": "none"},
                    ),
                    dbc.Row(
                        [
                            dbc.Col(html.P("Choose two companies to compare:"), md=12),
                            dbc.Col(
                                [
                                    dcc.Dropdown(
                                        id="bigrams-comp_1",
                                        options=[{"label": "Mean description length", "value": "mean_description"}, {"label": "Count movies", "value": "count_movies"}],
                                        value="count_movies",
                                    )
                                ],
                                md=6,
                            ),
                        ]
                    ),
                    dcc.Graph(id="bigrams-comps"),
                ],
                type="default",
            )
        ],
        style={"marginTop": 0, "marginBottom": 0},
    ),
]

BODY = dbc.Container(
    [
        dbc.Row([dbc.Col(dbc.Card(TOP_BIGRAM_COMPS)),], style={"marginTop": 30}),
        dbc.Row(
            [
                dbc.Col(LEFT_COLUMN, md=5, align="center"),
                dbc.Col(dbc.Card(TOP_BANKS_PLOT), md=7),
            ],
            style={"marginTop": 30},
        ),
        dbc.Row(
            [
                dbc.Col(LEFT_COLUMN2, md=5, align="center"),
                dbc.Col(dbc.Card(TOP_BANKS_PLOT2), md=7),
            ],
            style={"marginTop": 30},
        ),
        dbc.Card(WORDCLOUD_PLOTS),
        dbc.Row([dbc.Col([dbc.Card(LDA_PLOTS)])], style={"marginTop": 50}),
    ],
    className="mt-12",
)


app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
server = app.server  # for Heroku deployment

app.layout = html.Div(children=[NAVBAR, BODY])

"""
#  Callbacks
"""

@app.callback(
    Output("bigrams-comps", "figure"),
    [Input("bigrams-comp_1", "value")],
)
def comp_bigram_comparisons(comp_first):
    # change df_imdb to dataframe with only genres names and their mean description length
    description_length = imdb_df.groupby('genre')['description_length'].mean().reset_index()
    # change df_imdb to dataframe with only genres names and their sum of movies
    title = imdb_df.groupby('genre')['title'].count().reset_index()
    # merge description_length and title
    df_imdb_stats = pd.merge(description_length, title, on='genre')
    # add column dataset with imdb
    df_imdb_stats['dataset'] = 'imdb'

    # change df_imdb to dataframe with only genres names and their mean description length
    description_length = (filmweb_df.groupby('genre')['description_length'].mean() * (-1)).reset_index()
    # change df_imdb to dataframe with only genres names and their sum of movies
    title = (filmweb_df.groupby('genre')['title'].count() * (-1)).reset_index()
    # merge description_length and title
    df_filmweb_stats = pd.merge(description_length, title, on='genre')
    # add column dataset with imdb
    df_filmweb_stats['dataset'] = 'filmweb'

    # merge df_imdb_stats and df_filmweb_stats
    df_stats = pd.concat([df_imdb_stats, df_filmweb_stats])

    genres = df_stats.groupby('genre')['dataset'].count().reset_index()
    genres = genres[genres['dataset'] == 2]['genre'].tolist()
    df_stats = df_stats[df_stats['genre'].isin(genres)]

    if comp_first == 'mean_description':
        title = 'Mean description length'
        y = 'description_length'
    elif comp_first == 'count_movies':
        title = 'Number of movies'
        y = 'title'
    fig = px.bar(
        df_stats,
        title=title,
        x="genre",
        y=y,
        color="dataset",
        template="plotly_white",
        color_discrete_sequence=px.colors.qualitative.Bold,
        labels={"dataset": "Dataset:", "genre": "Genre"},
        hover_data="",
    )
    fig.update_layout(legend=dict(x=0.1, y=1.1), legend_orientation="h")
    fig.update_yaxes(title="", showticklabels=False)
    fig.data[0]["hovertemplate"] = fig.data[0]["hovertemplate"][:-14]
    return fig


@app.callback(
    [
        Output("time-window-slider", "marks"),
        Output("time-window-slider", "min"),
        Output("time-window-slider", "max"),
        Output("time-window-slider", "step"),
        Output("time-window-slider", "value"),
    ],
    [Input("bank-drop", "value")],
)
def populate_time_slider(value):
    min_date = datetime.strptime('1950', "%Y")
    max_date = datetime.strptime('2024', "%Y")

    marks = make_marks_time_slider(min_date, max_date)
    min_epoch = list(marks.keys())[0]
    max_epoch = list(marks.keys())[-1]

    return (
        marks,
        min_epoch,
        max_epoch,
        (max_epoch - min_epoch) / (len(list(marks.keys())) * 3),
        [min_epoch, max_epoch],
    )


@app.callback(
    Output("bank-drop", "options"),
    [Input("time-window-slider", "value")],
)
def populate_bank_dropdown(time_values):
    if time_values is not None:
        pass
    return [{"label": "IMDB", "value": "IMDB"}, {"label": "Filmweb", "value": "Filmweb"}]


@app.callback(
    Output("bank-drop22", "options"),
    [Input("time-window-slider22", "value")],
)
def populate_bank_dropdown2(time_values):
    if time_values is not None:
        pass
    return [{"label": "IMDB", "value": "imdb"}, {"label": "Filmweb", "value": "filmweb"}]

@app.callback(
    Output("bank-drop23", "options"),
    [Input("time-window-slider22", "value")],
)
def populate_bank_dropdown3(time_values):
    if time_values is not None:
        pass
    genres = ['action','adventure','animation','biography','comedy','crime','documentary','drama','family','fantasy','horror','music','musical','romance','sci-fi','short','thriller','war','western']
    dropdown = []
    for genre in genres:
        dropdown.append({"label": genre, "value": genre})
    return dropdown

@app.callback(
    Output("bank-drop2", "options"),
    [Input("time-window-slider", "value")],
)
def populate_bank_dropdown(time_values):
    if time_values is not None:
        pass
    return [{"label": "IMDB", "value": "IMDB"}, {"label": "Filmweb", "value": "Filmweb"}]


@app.callback(
    [Output("bank-sample", "figure"), Output("no-data-alert-bank", "style")],
    [Input("time-window-slider", "value"), Input("bank-drop", "value")],
)
def update_bank_sample_plot(time_values, bank_drop):
    print("redrawing bank-sample...")
    print("\ttime_values is:", time_values)
    print('BANK DROP', bank_drop)
    if time_values is None:
        return [{}, {"display": "block"}]

    min_date, max_date = time_slider_to_date(time_values)

    print('MIN AND MAX DATE', min_date, max_date)
    if bank_drop == 'IMDB':
        local_df = imdb_df
    elif bank_drop == 'Filmweb':
        local_df = filmweb_df
    local_df = local_df[(local_df.year >= int(min_date)) & (local_df.year <= int(max_date))]
    top_genres = local_df.genre.value_counts().head(10).index.tolist()
    top_genres_counts = local_df.genre.value_counts().head(10).tolist()

    data = [
        {
            "x": top_genres,
            "y": top_genres_counts,
            "text": top_genres,
            "textposition": "auto",
            "type": "bar",
            "name": "",
        }
    ]
    layout = {
        "autosize": False,
        "margin": dict(t=10, b=10, l=40, r=0, pad=4),
        "xaxis": {"showticklabels": False},
    }
    print("redrawing bank-sample...done")
    return [{"data": data, "layout": layout}, {"display": "none"}]


@app.callback(
    [
        Output("lda-table", "data"),
        Output("lda-table", "columns"),
        Output("tsne-lda", "figure"),
        Output("no-data-alert-lda", "style"),
    ],
    [Input("bank-drop2", "value"), Input("bank-drop3", "value"), Input("time-window-slider", "value")],
)
def update_lda_table(selected_bank, plot_option, time_values):
    lda_scatter_figure = populate_lda_scatter(selected_bank, plot_option)
    columns = [{'name': 'title', 'id': 'title'}, {'name': 'year', 'id': 'year'}, {'name': 'description', 'id': 'description'}, {'name': 'description_length', 'id': 'description_length'}, {'name': 'genre', 'id': 'genre'}]

    if selected_bank == 'Filmweb':
        local_df = filmweb_df[filmweb_df['year'].notna()]
    elif selected_bank == 'IMDB':
        local_df = imdb_df[imdb_df['year'].notna()]

    data = []
    for index, row in local_df.iterrows():
        item = {'title': row['title'], 'year': row['year'], 'description': row['description'], 'description_length': row['description_length'], 'genre': row['genre']}
        data.append(item)
    return (data, columns, lda_scatter_figure, {"display": "none"})


@app.callback(
    [
        Output("bank-wordcloud", "figure"),
        Output("frequency_figure", "figure"),
        Output("bank-treemap", "figure"),
        Output("no-data-alert", "style"),
    ],
    [
        Input("bank-drop", "value"),
        Input("time-window-slider", "value"),
        Input("bigrams-comp_3", "value"), 
        Input("bigrams-comp_4", "value"),
    ],
)
def update_wordcloud_plot(value_drop, time_values, dataset, genre):
    if dataset == 'IMDB':
        if genre == 'ALL':
            wordcloud, frequency_figure, treemap = plotly_wordcloud(imdb_df)
        else:
            wordcloud, frequency_figure, treemap = plotly_wordcloud(imdb_df[imdb_df.genre == genre])
    elif dataset == 'Filmweb':
        if genre == 'ALL':
            wordcloud, frequency_figure, treemap = plotly_wordcloud(filmweb_df)
        else:
            wordcloud, frequency_figure, treemap = plotly_wordcloud(filmweb_df[filmweb_df.genre == genre])

    alert_style = {"display": "none"}
    if (wordcloud == {}) or (frequency_figure == {}) or (treemap == {}):
        alert_style = {"display": "block"}
    return (wordcloud, frequency_figure, treemap, alert_style)

@app.callback(
    [Output("bank-sample22", "figure"), Output("no-data-alert-bank22", "style")],
    [Input("bank-drop22", "value"), Input("bank-drop23", "value")],
)
def update_bank_sample_plot(dataset, genre):
    x = pd.read_csv('./data_tfidf/' + genre + '_' + dataset +'.csv')['keyword'].tolist()
    y = pd.read_csv('./data_tfidf/' + genre + '_' + dataset +'.csv')['tf-idf'].tolist()
    data = [
        {
            "x": x,
            "y": y,
            "text": x,
            "textposition": "auto",
            "type": "bar",
            "name": "",
        }
    ]
    layout = {
        "autosize": False,
        "margin": dict(t=10, b=10, l=40, r=0, pad=4),
        "xaxis": {"showticklabels": False},
    }
    return [{"data": data, "layout": layout}, {"display": "none"}]

@app.callback(
    [Output("lda-table", "filter_query"), Output("lda-table-block", "style")],
    [Input("tsne-lda", "clickData")],
    [State("lda-table", "filter_query")],
)
def filter_table_on_scatter_click(tsne_click, current_filter):
    if tsne_click is not None:
        selected_complaint = tsne_click["points"][0]["hovertext"]
        if current_filter != "":
            filter_query = (
                "({title} eq "
                + str(selected_complaint)
                + ") || ("
                + current_filter
                + ")"
            )
        else:
            filter_query = "{title} eq " + str(selected_complaint)
        print("current_filter", current_filter)
        return (filter_query, {"display": "block"})
    return ["", {"display": "block"}]


@app.callback(Output("bank-drop", "value"), [Input("bank-sample", "clickData")])
def update_bank_drop_on_click(value):
    if value is not None:
        selected_bank = value["points"][0]["x"]
        return selected_bank
    return "IMDB"


if __name__ == "__main__":
    app.run_server(debug=True)
