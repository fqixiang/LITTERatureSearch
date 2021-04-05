# -*- coding: utf-8 -*-

# Run this app with `python app.py` and
# visit http://127.0.0.1:8050/ in your web browser.
# %%
import pandas as pd
import base64
import datetime
import io

# %%
import dash
import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output, State
import dash_table
from dash.exceptions import PreventUpdate

# %%
# external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__,
                external_stylesheets=[dbc.themes.BOOTSTRAP],
                prevent_initial_callbacks=True)

# %%
df = pd.read_csv('./data/data_example.csv')
key = df.Key[0]
title = df.Title[0]
abstract = df.Abstract[0]

# %%
# df.Title[df.Relevance.isna()]

# %%
markdown_text = '''
# **ML-assisted LITTERature Search**
    
#### A web application for finding literature on marine plastic pollution with the help of both human annotation and machine learning.
    '''

app.layout = html.Div(children=[
    html.Div(
        [dcc.Markdown(children=markdown_text)]),

    html.Label('===================================================================='),

    html.B('Step 1: Please upload your data set.'),
    html.Label('Instruction: you data set should have at least three columns - Title, Abstract and Relevance.'),

    # a button to upload your dataset
    html.Div([
        dcc.Upload(
            id='upload-data',
            children=html.Div([
                'Drag and Drop or ',
                html.A('Select Files')
            ]),
            style={
                'width': '80%',
                'height': '60px',
                'lineHeight': '60px',
                'borderWidth': '1px',
                'borderStyle': 'dashed',
                'borderRadius': '5px',
                'textAlign': 'center',
                'margin': '10px'
            },
            # Allow multiple files to be uploaded
            multiple=False
        )]),

    html.Div([html.Div(id='data_description_size'),
              html.Div(id='data_description_relevant'),
              html.Div(id='data_description_irrelevant'),
              html.Div(id='data_description_NA')]),

    html.Label('===================================================================='),

    # where the data is stored
    # Hidden div inside the app that stores the intermediate value
    html.Div(id='output-data-upload', style={'display': 'none'}),

    # step 2
    html.B('Step 2: Annotate more papers.'),

    html.Div(
        [html.Br(),
         html.B('New Title:'),
         html.Div(id='my-title'),
         html.B('New Abstract:'),
         html.Div(id='my-abstract')]),

    html.Label('===================================================================='),
    # annotate papers and show your progress
    html.Div([html.Label('Instruction: Enter 0 for irrelevant, 1 for relevant, 2 for uncertain.'),
              # html.Br(),
              dcc.Input(id='my-input', type='number', debounce=True,
                        placeholder='0, 1 or 2',
                        min=0, max=2, step=1, style={"width": "15%"}),
              html.Button(id='submit-button-state', n_clicks=0, children='Submit'),
              html.Div(id='my-count')
              ]),

    dcc.Store(id='memory_store'),
    html.Div(id='memory_count', style={'display': 'none'}),

    # step 3
    html.Label('===================================================================='),
    html.B('Step 3: Run model and check model performance. '),
    html.Label('Instruction: If happy, declare victory; otherwise, repeat step 2.'),
    html.Br(),

    # run model
    html.Div([html.Button(id='run_model',
                          n_clicks=0,
                          children='Run Model')])

],
    style={"width": "50%"})


# # call back for the data set
# @app.callback(Output('intermediate-value', 'children'), Input('dropdown', 'value'))
# def clean_data(value):
#      # some expensive clean data step
#      cleaned_df = your_expensive_clean_or_compute_step(value)
#
#      # more generally, this line would be
#      # json.dumps(cleaned_df)
#      return cleaned_df.to_json(date_format='iso', orient='split')


# function to parse uploaded data
def parse_contents(contents, filename, date):
    content_type, content_string = contents.split(',')

    decoded = base64.b64decode(content_string)
    try:
        if 'csv' in filename:
            # Assume that the user uploaded a CSV file
            df = pd.read_csv(
                io.StringIO(decoded.decode('utf-8')))
        elif 'xls' in filename:
            # Assume that the user uploaded an excel file
            df = pd.read_excel(io.BytesIO(decoded))
    except Exception as e:
        print(e)
        return html.Div([
            'There was an error processing this file.'
        ])

    return df.to_json(date_format='iso', orient='split')

    # return html.Div([
    #     html.B(children='Title:'),
    #     html.Div(children='{}'.format(df.Title[0])),
    #     html.B(children='Abstract:'),
    #     html.Div(children='{}'.format(df.Abstract[0]))
    # ])


# call back for saving the uploaded dataset
@app.callback(Output('output-data-upload', 'children'),
              Input('upload-data', 'contents'),
              State('upload-data', 'filename'),
              State('upload-data', 'last_modified'))
def save_uploaded_dataset(list_of_contents, list_of_names, list_of_dates):
    if list_of_contents is not None:
        data_jason = parse_contents(list_of_contents, list_of_names, list_of_dates)
        return data_jason


@app.callback(Output('data_description_size', 'children'),
              Output('data_description_relevant', 'children'),
              Output('data_description_irrelevant', 'children'),
              Output('data_description_NA', 'children'),
              Input('output-data-upload', 'children'))
def describe_dataset(jsonified_data):
    # more generally, this line would be
    # json.loads(jsonified_cleaned_data)
    if jsonified_data is not None:
        df = pd.read_json(jsonified_data, orient='split')

        n_rows = len(df)
        n_NA = df.Relevance.isna().sum()
        n_relevant = sum(df.Relevance == 1)
        n_irrelevant = sum(df.Relevance == 0)

        text_n_rows = 'There are {} papers in total.'.format(n_rows)
        text_n_relevant = '{} of them have been annotated as relevant.'.format(n_relevant)
        text_n_irrelevant = '{} of them have been annotated as irrelevant.'.format(n_irrelevant)
        text_n_NA = '{} still need to be annotated.'.format(n_NA)

        return text_n_rows, text_n_relevant, text_n_irrelevant, text_n_NA

    else:
        return '', '', '', ''


# call back for presenting a new title
@app.callback(
    Output(component_id='my-title', component_property='children'),
    Output(component_id='my-abstract', component_property='children'),
    Output(component_id='my-count', component_property='children'),
    Input('submit-button-state', 'n_clicks'),
    Input('output-data-upload', 'children'),
    State(component_id='my-input', component_property='value')
)
def update_output_title_div(n_clicks, jsonified_data, input_value):
    if jsonified_data is not None:
        df = pd.read_json(jsonified_data, orient='split')
        remaining_titles = df.Title[df.Relevance.isna()]
        remaining_abstracts = df.Abstract[df.Relevance.isna()]

        new_title = remaining_titles.iloc[n_clicks]
        new_abstract = remaining_abstracts.iloc[n_clicks]

    else:
        new_title = ''
        new_abstract = ''

    return new_title, new_abstract, 'You have annotated {} paper(s).'.format(n_clicks)


# save annotation
@app.callback(Output('memory_store', 'data'),
              Input('submit-button-state', 'n_clicks'),
              Input('my-input', 'value'),
              State('memory_store', 'data'))
def save_annotation(n_clicks, input_value, memory_store):
    if memory_store is None:
        annotation = ''

    else:
        annotation = memory_store
        annotation = annotation + '{}'.format(input_value)

    return annotation


# output the annotation data
@app.callback(Output('memory_count', 'children'),
              Input('memory_store', 'modified_timestamp'),
              State('memory_store', 'data'))
def show_annotation(timestamp, data):
    if timestamp is None:
        raise PreventUpdate

    data = data or {}

    return data


if __name__ == '__main__':
    app.run_server(debug=True)
