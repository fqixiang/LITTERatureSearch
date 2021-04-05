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
from model_functions import data_split_powerful, data_vectorizer_powerful, logistic_l1, find_highlight_word, feature_importance

# %%
# external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__,
                external_stylesheets=[dbc.themes.BOOTSTRAP],
                prevent_initial_callbacks=True)

# %% the ML model


# %%
card_title = [
    dbc.CardHeader("APP by Team MS @ComplexatonDataChallenge2020"),
    dbc.CardBody(
        [
            html.H1("ML-assisted LITTERature Search", className="card-title"),
            html.P(
                "This is a web application for quicker, easier literature search on marine plastic pollution with the help of both human annotation and machine learning.",
                className="card-text"),
        ]
    ),
]

upload_button = html.Div([
        dcc.Upload(
            id='upload-data',
            children=html.Div([
                'Drag and Drop or ',
                html.A('Select Files')
            ]),
            style={
                'width': '100%',
                'height': '60px',
                #'lineHeight': '10px',
                'borderWidth': '2px',
                'borderStyle': 'dashed',
                'borderRadius': '5px',
                'textAlign': 'center',
                'margin': 'auto',
                'left': '10px',
                'padding': '15px 0'
            },
            # Allow multiple files to be uploaded
            multiple=False
        )])

# card_step1 = [
#     dbc.CardBody(
#         [
#             html.H5("Step 1: Upload your data set and start annotation.", className="card-title"),
#             html.P(
#                 "You data set should have at least three columns - Title, Abstract and Relevance.",
#                 className="card-text",
#             ),
#         ]
#     ),
#     #upload_button
# ]

card_step1_1 = [
    dbc.CardBody(
        [
            html.H5("1. Upload your data set", className="card-title"),
            html.P(
                "Your data set (in csv or xls) should contain at least the following columns: Title, Keywords, Abstract, Journal, Link and Relevance.",
                className="card-text",
            ),
        ]
    )
]

card_step1_2 = [
    dbc.CardBody(
        [
            html.H5("2. Start annotation", className="card-title"),
            html.P(
                "Scan the title and abstract of a paper and decide whether it is relevant. Still not sure? "
                "Click the ''Full-text'' button for the full article.",
                className="card-text",
            ),
        ]
    )
]

card_step1_3 = [
    dbc.CardBody(
        [
            html.H5("3. Annotation Progress and Tools", className="card-title"),
            html.P(
                'Annotate as many paper as you like. When you are ready, go to "Step 2: Prediction Model". '
                'After that, two ML-based annotation tools will become available.',
                className="card-text")
        ]
    )
]

# card_step2 = [
#     dbc.CardBody(
#         [
#             html.H5("Step 2: Run your magic model.", className="card-title"),
#             html.P(
#                 "If you think you have annotated enough papers, you can try running the magic model and inspect its performance.",
#                 className="card-text",
#             ),
#         ]
#     ),
# ]

# card_step3 = [
#     dbc.CardBody(
#         [
#             html.H5("Step 3: Understand your model.", className="card-title"),
#             html.P(
#                 "Peek into the inner workings of your model and decide for yourself whether it makes sense.",
#                 className="card-text",
#             ),
#         ]
#     ),
# ]

new_title = dbc.Alert(
        [html.H5("Title"),
         dcc.Markdown(id='my-title')
        ])

new_abstract = dbc.Alert(
        [html.H5("Abstract"),
         dcc.Markdown(id='my-abstract')
        ])

new_link = dbc.Alert(
        [dbc.Badge("Full-text", id='my-link', color="secondary", className="mr-1", target='_blank'),
        ])

submit_button = dbc.Alert([dbc.FormGroup(
                            [
                                html.H5("This paper is ..."),
                                dbc.RadioItems(
                                    options=[
                                        {"label": "relevant", "value": 1},
                                        {"label": "irrelevant", "value": 0},
                                        {"label": "uncertain", "value": 2},
                                    ],
                                    #value=1,
                                    id="my-input",
                                    inline=False,
                                    switch=False,
                                ),
                            ]
                        ),

                          dbc.Button(id='submit-button-state', n_clicks=0, children='Submit', type='submit',
                                               style={'marginBottom': '0.8em'}),

                          html.Div(id='my-count'),

                          # below are all hidden
                          html.Div(id='memory_count',  style={'display': 'none'}
                                             ),
                          html.Div(id='my-key', style={'display': 'none'}
                                   ),
                          html.Div(id='key_count', style={'display': 'none'})])


switches = dbc.FormGroup(
    [

        #dbc.Badge("Explain", id='ml-assisted-badge', color="secondary", className="mr-1", target='_blank'),
        html.H5([html.Span("ML-assisted features", id='ml-assist-title')]),
        dbc.Tooltip(
            "Only available after running a model. "
            "Active sampling: papers that the model is most uncertain about will be shown first. <br>"
            "Highlight: words considered by the model useful for prediction would be highlighted."
            ,
            target="ml-assist-title",
        ),
        dbc.Checklist(
            options=[
                {"label": "Active sampling", "value": 1},
                {"label": "Highlight", "value": 2},
            ],
            value=[],
            id="switches-input",
            switch=True,
        ),
    ]
)

tab1_content = html.Div([
    dbc.Row(dbc.CardGroup([dbc.Card(card_step1_1, color="secondary", inverse=True, outline=False),
                           dbc.Card(card_step1_2, color="secondary", inverse=True, outline=False),
                           dbc.Card(card_step1_3, color="secondary", inverse=True, outline=False)])),



    dbc.Row([dbc.Col(html.Div([html.Br(),
                               upload_button,
                               html.Br(),
                               dbc.Alert([
                                   html.H4(id='data_upload_success'),
                                   html.Div(id='data_description_size'),
                                   html.Div(id='data_description_relevant'),
                                   html.Div(id='data_description_irrelevant'),
                                   html.Div(id='data_description_NA'),
                               ], id='data_summary',color='primary', is_open=False),]), width=4),
                  dbc.Col(html.Div([
                      html.Br(),
                      new_link,
                          new_title,
                          new_abstract,
                          ]), width=4),

                  # annotate papers and show your progress
                  dbc.Col(html.Div([html.Br(),
                                    submit_button,
                                    dbc.Alert([switches], id='ml-assisted', is_open=False),

                                    dbc.Label(id='switch-value', style={'display': 'none'}),
                                    #dbc.Label(id='uncertain-key-examples')
                   ]), width=4)]),

         html.Div(id='output-data-upload', style={'display': 'none'}),

        # all the necessary stored data are here
         dcc.Store(id='memory_store'),
         dcc.Store(id='memory_store_key'),
         dcc.Store(id='memory_store_key_uncertain'),
         dcc.Store(id='memory_word_ls')

         ],
#className="mt-3"
)


card_step2_1 = [
    dbc.CardBody(
        [
            html.H5("1. Run the prediction model.", className="card-title"),
            html.P(
                "Are you done with annotation or just want to try out the prediction model? Push the button below to run the model.",
                className="card-text",
            ),
        ]
    )
]

card_step2_2 = [
    dbc.CardBody(
        [
            html.H5("2. Model Performance.", className="card-title"),
            html.P(
                "Check out the performance of your model below. Not sure how to interpret? Click here for explanation.",
                className="card-text",
            ),
        ]
    )
]

card_step2_3 = [
    dbc.CardBody(
        [
            html.H5("3. What next?", className="card-title"),
            html.P(
                "Would you like to annotate more? Save the current results? Or are you happy and would like to download the final predictions?",
                className="card-text",
            ),
        ]
    )
]


model_alert = dbc.Alert(
    'Model running... This will take just a few seconds!',
    id='model-alert',
    is_open=False,
    color='primary'
)

model_performance = html.Div(
    [
        dbc.Alert(id='baseline', color="primary"),
        dbc.Alert(id='accuracy', color="secondary"),
        dbc.Alert(id='balanced-accuracy', color="success"),
        dbc.Alert(id='precision', color="warning"),
        dbc.Alert(id='recall', color="danger"),
        dbc.Alert(id='f1', color="info"),
        dbc.Alert(id='auc', color="dark")
    ]
)


tab2_content = html.Div([
    dbc.Row(dbc.CardGroup([dbc.Card(card_step2_1, color="secondary", inverse=True, outline=False),
                           dbc.Card(card_step2_2, color="secondary", inverse=True, outline=False),
                           dbc.Card(card_step2_3, color="secondary", inverse=True, outline=False)])),

    dbc.Row([dbc.Col(html.Div([html.Br(),
                      dbc.Button(["Click to Run Prediction Model"],
                                 color="primary",
                                 outline=False,
                                 block=True,
                                 n_clicks=0,
                                 id='run_model'),
                               #dbc.Label(id='model_result'),
                               model_alert
                      ]), width=4),

             dbc.Col(html.Div([html.Br(),
                               model_performance,
                               ]), width=4),

             dbc.Col(html.Div([html.Br(),
                      dbc.Button("I would like to annotate more!", color="success", className="mr-1", block=True, disabled=False, id='choice1-annotate'),
                      dbc.Button("I would like to save the annotated results.", color="success", active=True, className="mr-1", block=True, disabled=False, id='choice2-save'),
                      dbc.Button("I am happy and would like to save the prediction results.", color="success", disabled=False, block=True, id='choice3-done'),
                      html.Br(),
                      dbc.Alert('''Great that you would like to annotate more papers! This time, however, 
                      you can speed things up by using two ML-assisted features for annotation. Go back to Step 1 and you should see the options available now.''',
                                is_open=False, id='choice1-alert')
                      ]), width=4)])
])



toast_feature_importance = html.Div(
    [
        dbc.Button(
            "What is this?",
            id="toast_feature_importance",
            color="secondary",
            className="mb-3",
            block=True
        ),
        dbc.Modal(
            [
                dbc.ModalHeader("Permutation Feature Importance"),
                dbc.ModalBody([html.P('''Permutation feature importance scores are used to evaluate how important a feature is 
                in the predictive performance of a model. The higher the score is, the more important a feature is. A
                score equal to or below zero indicates that the feature is not important at all. 
                '''),
                               html.P('''
                In this figure, you can see the most important features (i.e. words), defined as having a score > 0.01, in our prediction model. 
                This can help you understand which features (i.e. words) the model uses and how they relate to the prediction
                goal. 
                '''),
                               html.P('''The very same "important" words are used by the "Highlight" feature (in Step 1) to help you speed up your annotation.
                                               ''')
                               ]
                    ),
                dbc.ModalFooter(
                    dbc.Button(
                        "Close", id="close-centered", className="ml-auto"
                    )
                ),
            ],
            id="modal-centered",
            centered=True,
        ),
    ]
)

tab3_content = html.Div([
    dbc.Row([dbc.Col(toast_feature_importance, width=12)]),

    dbc.Row([dcc.Graph(id='feature_importance', responsive=True, style={'width': '200vh', 'height': '100vh', "margin-left": "auto",
            "margin-right": "auto",})])])



app.layout = html.Div(children=[
    dbc.Row([dbc.Col(dbc.Card(card_title, color="primary", inverse=True))]),

    dbc.Tabs(
        [   dbc.Tab(tab1_content, label="Step 1: Annotation"),
            dbc.Tab(tab2_content, label="Step 2: Prediction Model"),
            dbc.Tab(tab3_content, label="Bonus: Model Explanation", disabled=False, id='tab3'
            ),
        ], id='tabs'
    ),
    html.Div(id="tab-content"),

])


# @app.callback(
#     Output("tab-content", "children"),
#     [Input("tabs", "active_tab")],
# )
# def tab_content(active_tab):
#     if active_tab == "Bonus: Model Explanation":
#         return tab3_content


# function to parse uploaded data
def parse_contents(contents, filename, date):
    content_type, content_string = contents.split(',')

    decoded = base64.b64decode(content_string)
    try:
        if 'csv' in filename:
            # Assume that the user uploaded a CSV file
            df = pd.read_csv(io.StringIO(decoded.decode('utf-8')))
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
              Output('data_upload_success', 'children'),
              Output('data_summary', 'is_open'),
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

        success_message = 'Your data is uploaded successfully!'

        return text_n_rows, text_n_relevant, text_n_irrelevant, text_n_NA, success_message, True

    else:
        return '', '', '', ''


# call back for presenting a new title
@app.callback(
    Output(component_id='my-title', component_property='children'),
    Output(component_id='my-abstract', component_property='children'),
    Output(component_id='my-link', component_property='href'),
    Output(component_id='my-count', component_property='children'),
    Output(component_id='my-key', component_property='children'),
    Output('switch-value', 'children'),
    Input('submit-button-state', 'n_clicks'),
    Input('output-data-upload', 'children'),
    Input('switches-input', 'value'),
    State(component_id='memory_store', component_property='data'),
    State(component_id='memory_store_key_uncertain', component_property='data'),
    State('memory_word_ls', 'data')
)
def update_output_title_div(n_clicks, jsonified_data, switch, annotation_data, key_uncertain, word_ls):
    # pretend this is the important word list
    feature_ls = ['is', 'the']

    # when we upload data and annotate first time
    if jsonified_data is not None and (len(switch) == 0) or (sum(switch) == 2):
        df = pd.read_json(jsonified_data, orient='split')
        remaining_titles = df.Title[df.Relevance.isna()]
        remaining_abstracts = df.Abstract[df.Relevance.isna()]
        remaining_link = df.Link[df.Relevance.isna()]
        remaining_keys = df.Key[df.Relevance.isna()]

        new_title = remaining_titles.iloc[n_clicks]
        new_abstract = remaining_abstracts.iloc[n_clicks]
        new_link = remaining_link.iloc[n_clicks]
        new_key = remaining_keys.iloc[n_clicks]
        switch_value = 'no machine assisted annotation'

        if sum(switch) == 2:
            # get the important words from permutation importance algorithm
            word_ls = word_ls.split()

            # highlight title and abstract
            new_title = find_highlight_word(word_ls, new_title)
            new_abstract = find_highlight_word(word_ls, new_abstract)

            # just to see what's going on
            switch_value = 'regular + highlight'

    # if there is active sampling involved
    elif jsonified_data is not None and (1 in switch):
        # get the stored prediction results
        key_uncertain_ls = key_uncertain.split()
        #key_uncertain_ls = '3D5HUXSL KE2NTCS7 BELAGTQW'.split()
        # start new n_click count (subtract by the number of saved annotations, probably)
        df = pd.read_json(jsonified_data, orient='split')

        new_key = key_uncertain_ls[n_clicks]
        new_title = df.Title[df.Key == new_key].iloc[0]
        new_abstract = df.Abstract[df.Key == new_key].iloc[0]
        new_link = df.Link[df.Key == new_key].iloc[0]

        switch_value = 'only active sampling'

        if 2 in switch:
            # get the important words from permutation importance algorithm
            word_ls = word_ls.split()

            # highlight title and abstract
            new_title = find_highlight_word(word_ls, new_title)
            new_abstract = find_highlight_word(word_ls, new_abstract)

            # just checking
            switch_value = 'both active sampling and highlight'

    else:
        new_title = ''
        new_abstract = ''
        new_link = ''
        new_key = ''
        switch_value = 'none'

    return new_title, new_abstract, new_link, 'You have annotated additional {} paper(s).'.format(n_clicks), new_key, switch_value


# save annotation
@app.callback(Output('memory_store', 'data'),
              Output('memory_store_key', 'data'),
              Input('submit-button-state', 'n_clicks'),
              State('my-key', 'children'),
              State('my-input', 'value'),
              State('memory_store_key', 'data'),
              State('memory_store', 'data'))
def save_annotation(n_clicks, key_value, input_value,  memory_store_key, memory_store_annotation):

    if memory_store_key is None or (memory_store_annotation is None):
        annotation = '{}'.format(input_value)
        key_annotation = '{}'.format(key_value)

    else:
        annotation = memory_store_annotation
        key_annotation = memory_store_key

        if n_clicks > len(annotation):
            annotation = annotation + '{}'.format(input_value)
            key_annotation = key_annotation + ' ' + '{}'.format(key_value)

    return annotation, key_annotation


# output the annotation data
@app.callback(Output('memory_count', 'children'),
              Output('key_count', 'children'),
              Input('memory_store', 'modified_timestamp'),
              State('memory_store', 'data'),
              State('memory_store_key', 'data'))
def show_annotation(timestamp, annotation_data, key_data):
    if timestamp is None:
        raise PreventUpdate

    annotation_data = annotation_data or {}
    key_data = key_data or {}

    return annotation_data, key_data


# put up model running alert
@app.callback(Output('model-alert', 'is_open'),
              Input('run_model', 'n_clicks'))
def model_wait_alert(n_clicks):
    if n_clicks > 0:
        return True


# run a simple model on the annotated data
@app.callback(Output('model-alert', 'children'),
              Output('baseline', 'children'),
              Output('accuracy', 'children'),
              Output('balanced-accuracy', 'children'),
              Output('precision', 'children'),
              Output('recall', 'children'),
              Output('f1', 'children'),
              Output('auc', 'children'),
              Output('memory_store_key_uncertain', 'data'),
              Output('memory_word_ls', 'data'),
              Output('feature_importance', 'figure'),
              Input('run_model', 'n_clicks'),
              State('output-data-upload', 'children'),
              State('memory_store', 'data'),
              State('memory_store_key', 'data'))
def run_model(n_click, jsonified_data, annotation, annotated_keys):
    if n_click > 0 and (jsonified_data is not None):
        df = pd.read_json(jsonified_data, orient='split')

        if annotation is not None:
            annotation_ls = [int(integer) for integer in annotation]
            key_ls = annotated_keys.split()
            df_annotation = pd.DataFrame({'Key': key_ls,
                                          'Relevance': annotation_ls})

            # n rows with na before
            n_row_annotated_old = df.Relevance.isna().sum()

            # update with new annotations
            df.update(df.drop('Relevance', axis=1).merge(df_annotation, 'left', 'Key'))
            n_row_annotated_new = df.Relevance.isna().sum()

        # data preprocessing
        dataset_train, dataset_test, dataset_new = data_split_powerful(df, seed=123, test_size=.20)

        x_train, y_train, x_test, y_test, x_new = data_vectorizer_powerful(dataset_train, dataset_test, dataset_new,
                                                           var=['Title', 'Abstract', 'Keywords', 'Venue'],
                                                           n_features=100)

        # train lasso model
        results = logistic_l1(x_train, y_train, x_test, y_test, scoring="f1")

        baseline = "Baseline accuracy: {:.3f}".format(y_test.mean())
        accuracy = "Accuracy: {:.3f}".format(results['accuracy'])
        balanced_acc = "Balanced accuracy: {:.3f}".format(results['balanced_acc'])
        precision = "Precision: {:.3f}".format(results['precision'])
        recall = "Recall: {:.3f}".format(results['recall'])
        f1 = "F1: {:.3f}".format(results['f1'])
        auc = "AUC: {:.3f}".format(results['auc'])

        model_finished_alert = 'Done! Time to check out the performance of the model.'

        # predictions on the remaining data
        new_prob = results['model'].predict_proba(x_new)[:, 1]
        new_key = dataset_new.Key.to_list()

        df_prediction = pd.DataFrame({'Key': new_key, 'Prob': new_prob})
        df_prediction = df_prediction.assign(Certainty=abs(df_prediction.Prob - 0.5))
        new_keys = df_prediction.sort_values(by=['Certainty'], ascending=True).Key.to_list()
        new_keys = ' '.join(key for key in new_keys)

        # feature importance
        word_ls, feature_fig = feature_importance(model=results['model'],
                                                  x=x_train,
                                                  y=y_train,
                                                  threshold=0.01)

        word_ls = ' '.join(word for word in word_ls)

        return model_finished_alert, baseline, accuracy, balanced_acc, precision, recall, f1, auc, new_keys, word_ls, feature_fig

    else:
        return None, None, None, None, None, None, None, None, None

# put up model running alert
@app.callback(Output('choice1-alert', 'is_open'),
              Output('ml-assisted', 'is_open'),
              Input('choice1-annotate', 'n_clicks'))
def choice1_alert(n_clicks):
    if n_clicks > 0:
        return True, True


@app.callback(
    Output("modal-centered", "is_open"),
    [Input("toast_feature_importance", "n_clicks"), Input("close-centered", "n_clicks")],
    [State("modal-centered", "is_open")],
)
def toggle_modal(n1, n2, is_open):
    if n1 or n2:
        return not is_open
    return is_open



if __name__ == '__main__':
    app.run_server(debug=True)
