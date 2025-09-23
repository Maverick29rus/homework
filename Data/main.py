import os
import io
import streamlit as st
import streamlit.components.v1 as components
import ydata_profiling
from streamlit_pandas_profiling import st_profile_report
import numpy as np
import pandas as pd
import plotly.express as px
from sklearn.preprocessing import (
    LabelEncoder,
    OrdinalEncoder,
    OneHotEncoder,
    StandardScaler,
    RobustScaler,
    QuantileTransformer,
    MinMaxScaler,
)
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from keras.models import Sequential
from keras.layers import Dense, Dropout, Input
from keras.regularizers import l2
from keras.callbacks import EarlyStopping
from keras.saving import load_model
from keras.optimizers import Adam, SGD, RMSprop, Adagrad, Adadelta, Nadam
from keras.losses import CategoricalCrossentropy, SparseCategoricalCrossentropy
from keras.utils import clear_session
# pip install kagglehub[pandas-datasets]
import kagglehub
from kagglehub import KaggleDatasetAdapter

# Function for downloading dataset from Kaggle
def load_dataset():
    df = kagglehub.dataset_load(
        KaggleDatasetAdapter.PANDAS,
        'milapgohil/flavorsense-tastes-predicted-by-life-and-climate',
        'FlavorSense.csv',
    )
    df.to_csv('FlavorSense.csv', index=False)
    return df

# Check for a local file, if not, download from Kaggle
if os.path.exists('FlavorSense.csv'):
    df = pd.read_csv('FlavorSense.csv')
else:
    df = load_dataset()

# Defining lists of numerical and categorical features
numeric_features = ['age']
categorical_features = [
    'sleep_cycle',
    'exercise_habits',
    'climate_zone',
    'historical_cuisine_exposure',
]

# Functions for updating the session status
def model_trained(state):
    st.session_state.model_trained = state
def preprocess_data(state):
    st.session_state.preprocess = state

# Initializing the session state
if 'model_trained' not in st.session_state:
    st.session_state.model_trained = False
if 'model_saved' not in st.session_state:
    st.session_state.model_saved = False

st.title('FlavorSense: Tastes Predicted by Life & Climate')

# Selecting the application's operating mode
app_mode = st.sidebar.radio(
    'Page',
    [
        'About',
        'EDA',
        'Preparation',
        'Modeling',
        'Prediction',

    ],
    captions=[
        'About this app',
        'Data exploration and visualization',
        'Data preparation',
        'Building and training model',
        'Prediction on you data',
    ],
)

# Handling the switch display of duplicates and missing values
if app_mode == 'EDA' or app_mode == 'Preparation':
    st.session_state.drop_duplicates = st.sidebar.checkbox('Drop duplicates', on_change=preprocess_data, args=[False])
    st.session_state.drop_missing = st.sidebar.checkbox('Drop missing values', on_change=preprocess_data, args=[False])

# Removing duplicates and missing values
if st.session_state.get('drop_duplicates'):
    df.drop_duplicates(inplace=True)
if st.session_state.get('drop_missing'):
    df.dropna(inplace=True)

# Checking whether preprocessing needs to be performed
if st.session_state.get('preprocess'):

    # Choosing a scaler for numeric features
    if st.session_state.scaler_name == 'Standard Scaler':
        scaler = StandardScaler()
    elif st.session_state.scaler_name == 'Robust Scaler':
        scaler = RobustScaler()
    elif st.session_state.scaler_name == 'Quantile Transformer':
        scaler = QuantileTransformer()
    else:
        scaler = MinMaxScaler()

    # Choosing an encoder for categorical features
    if st.session_state.encoder_name == 'Ordinal Encoder':
        encoder = OrdinalEncoder(encoded_missing_value=-1)
    else:
        encoder = OneHotEncoder(
            handle_unknown='ignore', drop='first', sparse_output=False
        )

    # Selecting the encoder for the target variable
    if st.session_state.target_encoder_name == 'Label Encoder':
        target_encoder = LabelEncoder()
        y = target_encoder.fit_transform(df['preferred_taste'])
        st.session_state.n_categories = len(target_encoder.classes_)
        st.session_state.labels = list(target_encoder.classes_)
        st.session_state.loss = 'sparse_categorical_crossentropy'
    else:
        target_encoder = OneHotEncoder(
            handle_unknown='ignore', sparse_output=False
        )
        y = target_encoder.fit_transform(df[['preferred_taste']])
        st.session_state.n_categories = y.shape[1]
        st.session_state.labels = target_encoder.categories_[0]
        st.session_state.loss = 'categorical_crossentropy'

    # Handling missing values
    if st.session_state.drop_missing == False:
        if st.session_state.num_fill_name == 'Mean':
            df['age'] = df['age'].fillna(df['age'].mean())
        elif st.session_state.num_fill_name == 'Median':
            df['age'] = df['age'].fillna(df['age'].median())
        else:
            df['age'] = df['age'].fillna(st.session_state.fillna_number)

    # Scaling of numerical features
    Num = scaler.fit_transform(df[numeric_features])

    # Categorical feature encoding
    Cat = encoder.fit_transform(df[categorical_features])

    # Combining numerical and categorical features
    X = np.hstack((Cat, Num))

    # Splitting the data into training and test samples
    X_train, X_test, y_train, y_test = train_test_split(
            X, 
            y, 
            stratify=y, 
            test_size=0.2, 
            random_state=42,
    )
    st.session_state.x_shape = X_train.shape[1]

if app_mode == 'About':
    st.markdown(
        'Classify food tastes using life routines, geography & upbringing cuisine'
    )
    st.text(
        '''About Dataset

        Dive into a unique synthetic dataset that predicts food taste preferences—Spicy,  
        Sweet, Sour, or Salty—based on lifestyle habits, sleep cycles, exercise levels,  
        climate zones, and cultural cuisine exposure. Perfect for classification projects,  
        EDA practice, or building recommendation systems with a flavorful twist!

        1.Multi-class Classification: Predict preferred_taste using all features.

        2.EDA & Visualization: Explore how lifestyle and climate impact food preferences.

        3.Missing Value Handling: Practice imputation techniques (SimpleImputer, KNN, etc.)

        4.Cultural Feature Encoding: Encode historical_cuisine creatively (OneHot, Embedding).

        5.Model Comparison: Test Decision Trees, SVM, XGBoost, or Neural Nets.

        6.Recommendation System: Build a basic taste preference recommender.'''
    )

    st.markdown(
        'Original data taken from \
        [Kaggle](https://www.kaggle.com/datasets/milapgohil/flavorsense-tastes-predicted-by-life-and-climate/data)'
    )

elif app_mode == 'EDA':
    # Getting a list of all the columns in the DataFrame
    cols = df.columns.to_list()
    # User Instructions
    st.markdown('Tick the toggle on the side panel to explore the dataset.')

    if st.toggle('Show overview'):
        # Creating a profile report using pandas-profiling
        pr = df.profile_report()
        st_profile_report(pr)

    # Displaying the DataFrame in Streamlit
    if st.toggle('Show Dataset'):
        st.dataframe(df)

    # Displaying "Dataset info"
    if st.toggle('Dataset info'):
        # Using io.StringIO to capture the output df.info ()
        buffer = io.StringIO()
        df.info(buf=buffer)
        s = buffer.getvalue()
        st.code(s)

    # Calculate and display the percentage of missing values in each column
    if st.toggle('Missing Values?'):
        st.subheader('Checking for missing values in each columns ( in % )')
        st.write(df.isnull().mean() * 100)

    # Calculate and display the number of duplicates
    if st.toggle('Checking for dublicates'):
        st.write(f'Total dublicates: {df.duplicated().sum()}')

    # Visualization
    # Function for plotting a pie chart
    def create_pie(
        df,
        col,
        title=None,
        hole=0,
        width=500,
        height=500,
        show_legend=False,
        textinfo='percent+label',
    ):
        # Setting the header if it has not been transmitted
        if title is None:
            col_name = col.replace('_', ' ').title()
            title = col_name + ' Distribution'

        fig = px.pie(
            df,
            names=col,
            title=title,
            width=width,
            height=height,
            hole=hole,
        )

        fig.update_layout(title_x=0.5)
        fig.update_traces(textinfo=textinfo, showlegend=show_legend)
        st.plotly_chart(fig, use_container_width=False, theme=None)
    # Function for plotting a histogram
    def create_histogram(
        df,
        x,
        hue=None,
        barmode='group',
        nbins=None,
        nticks=None,
        width=1000,
        height=500,
    ):
        x_label = x.replace('_', ' ').title()
        if hue:
            hue_label = hue.replace('_', ' ').title()
            labels = {hue: hue_label, x: x_label}
            title_text = f'Count Plot for {x_label} by {hue_label}'
        else:
            labels = {x: x_label}
            title_text = f'Count Plot for {x_label}'
        col_order = {
            'preferred_taste': ['Sweet', 'Salty', 'Sour'],
            'sleep_cycle': ['Early Bird', 'Irregular', 'Night Owl'],
            'exercise_habits': ['Light', 'Moderate', 'Heavy'],
            'climate_zone': ['Cold', 'Temperate', 'Tropical', 'Dry'],
            'historical_cuisine_exposure': ['Mediterranean', 'Mixed', 'Asian'],
        }
        fig = px.histogram(
            df,
            x=x,
            color=hue,
            barmode=barmode,
            nbins=nbins,
            width=width,
            height=height,
            category_orders=col_order,
            labels=labels,
        )

        fig.update_layout(
            title_text=title_text,
            yaxis_title='Count',
            title_x=0.5,
            bargap=0.2,
        )

        fig.update_xaxes(
            nticks=nticks,
            showgrid=True,
            gridwidth=1,
            gridcolor='LightGray',
        )

        fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='LightGray')
        st.plotly_chart(fig, use_container_width=False, theme=None)
    # Function for plotting a box plot
    def create_box(
        df,
        x,
        y,
        title=None,
        x_label=None,
        y_label=None,
        color=None,
        boxmode='group',
        width=800,
        height=600,
        horizontal=True,
    ):
        # Setting default names if they are not specified
        x_name = x.replace('_', ' ').title()
        y_name = y.replace('_', ' ').title()
        # Setting the chart title if it is not set
        if title is None:
            if horizontal:
                title = f'{x_name} Distribution Across {y_name}'
            else:
                title = f'{y_name} Distribution Across {x_name}'
        # Setting axis labels if they are not specified
        if x_label is None:
            x_label = x_name
        if y_label is None:
            y_label = y_name

        fig = px.box(
            df,
            x=x,
            y=y,
            color=color,
            title=title,
            # width=width,
            # height=height,
            boxmode=boxmode,
        )

        fig.update_layout(xaxis_title=x_label, yaxis_title=y_label)
        fig.update_traces(showlegend=False)

        st.plotly_chart(fig, theme=None)

    # Enabling the display of categorical data distribution
    if st.toggle('Categorical data distribution'):
        st.subheader('Categorical data distribution')
        tab1, tab2, tab3, tab4, tab5 = st.tabs(cols[1:6])
        with tab1:
            create_pie(df, cols[1])
        with tab2:
            create_pie(df, cols[2])
        with tab3:
            create_pie(df, cols[3])
        with tab4:
            create_pie(df, cols[4])
        with tab5:
            create_pie(df, cols[5])

    # Enabling the display of categorical data in relation to the target variable
    if st.toggle('Categorical vs target'):
        st.subheader('Categorical vs target')
        tab1, tab2, tab3, tab4 = st.tabs(cols[1:5])
        with tab1:
            create_histogram(df, cols[1], cols[-1])
        with tab2:
            create_histogram(df, cols[2], cols[-1])
        with tab3:
            create_histogram(df, cols[3], cols[-1])
        with tab4:
            create_histogram(df, cols[4], cols[-1])

    # Enabling the age distribution display
    if st.toggle('Age Distribution'):
        st.subheader('Age Distribution')
        create_histogram(df, cols[0], nbins=30, nticks=11)

    # Enabling the display of age relative to the target variable
    if st.toggle('Age vs target'):
        st.subheader('Age vs target')
        create_box(df, cols[0], cols[-1])

elif app_mode == 'Preparation':
    st.session_state.model_trained = False
    st.session_state.model_saved = False

    st.header('Data preparation')
    st.write('Please choose parameters and click the button on the sidepanel')
    st.subheader('Prepare feature')
    # Handling missing values
    if st.session_state.drop_missing == False:
        st.subheader('Handling missing values')
        st.session_state.num_fill_name = st.selectbox(
            'Choose age fill missing metod:',
            ('Mean', 'Median', 'Number'),
            index=1,
        )
        if st.session_state.num_fill_name == 'Number':
            st.session_state.fillna_number = st.number_input(
                'Choose a number',
                min_value=int(df['age'].min()),
                max_value=int(df['age'].max()),
                value=int(df['age'].median()),
                step=1,
            )

    # Choosing a scaler
    st.subheader('Choosing a scaler')
    st.session_state.scaler_name = st.selectbox(
        'Choose numeric feature scaler:',
        ('Standard Scaler', 'Robust Scaler', 'Quantile Transformer', 'Min-Max Scaler'),
        index=0,
    )
    # Choosing a encoder
    st.subheader('Choosing a encoder')
    st.session_state.encoder_name = st.selectbox(
        'Choose categorical feature encoder:',
        ('Ordinal Encoder', 'One Hot Encoder'),
        index=1,
    )
    # Choosing a target encoder
    st.subheader('Prepare target')
    st.session_state.target_encoder_name = st.selectbox(
        'Choose target encoder:',
        ('Label Encoder', 'One Hot Encoder'),
        index=0,
    )
    # Button that informs the rest of the program that the data can be preprocessed
    if st.sidebar.button('Preprocess Data', on_click=preprocess_data, args=[True]):
        # Print samples shape
        st.write(f'Train samples shape: {X_train.shape}')
        st.write(f'Test samples shape: {X_test.shape}')
        st.write(f'Train targets shape: {y_train.shape}')
        st.write(f'Test targets shape: {y_test.shape}')
        st.write('Lets look at first sample:')
        st.text(X[0].tolist())
        st.write('Lets look at first 5 targets:')
        st.text(y[:5].tolist())

    # Button to reset the data preprocessing
    st.sidebar.button('Clear Data', on_click=preprocess_data, args=[False])  

elif app_mode == 'Modeling':
    if st.session_state.get('preprocess'):
        # Building model
        st.header('Modeling')
        st.write('Please choose parameters and click the button on the sidepanel')
        print([st.session_state.get('model_trained')])
        def build_model_user(
            INPUT=10,
            LAYERS=[8, 8, 8],
            ACTIVATION='relu',
            FINAL_ACTIVATION='softmax',
            OPTIMIZER='Adam',
            LEARNING_RATE=0.003,
            LOSS='sparse_categorical_crossentropy',
            DROPOUT_RATE=0.0,
            REGULARIZATION=0.0,
            N_CATEGORIES=4,
        ):
            model = Sequential()
            model.add(Input(shape=(INPUT,)))
            # Set the number of layers, their dimensions and activation functions
            for units in LAYERS:
                model.add(
                    Dense(
                        units=units,
                        activation=ACTIVATION,
                        kernel_regularizer=(
                            l2(REGULARIZATION) if REGULARIZATION > 0 else None
                        ),
                    )
                )
            # Add a Dropout layer
            if DROPOUT_RATE > 0:
                model.add(Dropout(DROPOUT_RATE))

            # Add a final layer
            model.add(Dense(N_CATEGORIES, activation=FINAL_ACTIVATION))

            # Choosing an optimizer
            if OPTIMIZER == 'Adam':
                opt = Adam(learning_rate=LEARNING_RATE)
            elif OPTIMIZER == 'SGD':
                opt = SGD(learning_rate=LEARNING_RATE)
            elif OPTIMIZER == 'RMSprop':
                opt = RMSprop(learning_rate=LEARNING_RATE)
            elif OPTIMIZER == 'Adagrad':
                opt = Adagrad(learning_rate=LEARNING_RATE)
            elif OPTIMIZER == 'Adadelta':
                opt = Adadelta(learning_rate=LEARNING_RATE)
            else:
                opt = Nadam(learning_rate=LEARNING_RATE)

            # Selection of the loss function
            if LOSS == 'sparse_categorical_crossentropy':
                loss_fn = SparseCategoricalCrossentropy()
            else:
                loss_fn = CategoricalCrossentropy()

           # Compilation of the model
            model.compile(optimizer=opt, loss=loss_fn, metrics=['accuracy'])

            return model
        # Initializing the model parameters
        if st.session_state.model_trained == False:
            st.subheader('Building model')

            st.session_state.num_layers = st.number_input(
                'Choose a number of layers',
                min_value=1,
                max_value=10,
                value=3,
                step=1,
                placeholder='Type a number between 1 and 10',
            )

            st.session_state.layers = []
            for layer in range(1, st.session_state.num_layers + 1):
                num_units = st.number_input(
                    f'Choose a number of units for {layer} layer',
                    min_value=1,
                    max_value=1024,
                    value=8,
                    step=1,
                    placeholder='Type a number between 1 and 1024'
                )
                st.session_state.layers.append(num_units)

            st.session_state.activation = st.selectbox(
                'Choose a activation function for hidden layers:',
                ('relu', 'sigmoid', 'tanh', 'softmax', 'leaky_relu', 'elu', 'swish'),
                index=0,
            )

            st.session_state.dropout_rate = st.number_input(
                'Choose a dropout rate (0 - no dropout)',
                min_value=0.0,
                max_value=0.7,
                step=0.01,
                placeholder='Type a number between 0 and 0.7',
            )

            st.session_state.regularization = st.number_input(
                'Choose a l2 regularization rate (0 - no regularization)',
                min_value=0.0,
                max_value=10.0,
                step=0.01,
                placeholder='Type a number between 0 and 10',
            )

            st.session_state.final_activation = st.selectbox(
                'Choose a activation function for final layer:',
                ('softmax', 'sigmoid', 'tanh', 'relu'),
                index=0,
            )

            st.session_state.optimizer = st.selectbox(
                'Choose a optimizer:',
                ('Adam', 'SGD', 'RMSprop', 'Adagrad', 'Adadelta', 'Nadam'),
                index=0,
            )

            st.session_state.learning_rate = st.number_input(
                'Choose a learning rate',
                min_value=1e-6,
                max_value=1.0,
                value=1e-3,
                step=1e-6,
                format='%0.6f',
                placeholder='Type a number between 0,000001 and 1',
            )
        # Creating a model or load saved model
        if st.session_state.model_saved == False:
            model = build_model_user(
                INPUT=st.session_state.x_shape,
                LAYERS=st.session_state.layers,
                ACTIVATION=st.session_state.activation,
                FINAL_ACTIVATION=st.session_state.final_activation,
                OPTIMIZER=st.session_state.optimizer,
                LEARNING_RATE=st.session_state.learning_rate,
                LOSS=st.session_state.loss,
                DROPOUT_RATE=st.session_state.dropout_rate,
                REGULARIZATION=st.session_state.regularization,
                N_CATEGORIES=st.session_state.n_categories,
            )
        else:
            model = load_model('flavorsense_model.keras')

        model.summary(print_fn=lambda x: st.text(x))

        # Training model
        st.subheader('Training model')

        # Function for training and saving the model
        def train_model(
            MODEL,
            X_TRAIN,
            Y_TRAIN,
            BATCH_SIZE=16,
            EPOCHS=50,
            VALIDATION_SPLIT=0.2,
            EARLY_STOP=False,
            PATIENCE=10,
            MIN_DELTA=0.001,
        ):
            '''
            Функция для обучения модели с использованием ранней остановки.

            :param model: Модель, которую нужно обучить.
            :param X_train: Данные для обучения.
            :param y_train: Метки для обучения.
            :param batch_size: Размер пакета для обучения.
            :param epochs: Количество эпох для обучения.
            :param validation_split: Доля данных для валидации.
            :param patience: Количество эпох без улучшений для ранней остановки.
            :param min_delta: Минимальное изменение для учета улучшения.
            :return: История обучения модели и сама модель
            '''
            if EARLY_STOP:
                early_stop = EarlyStopping(
                    monitor='val_accuracy',
                    min_delta=MIN_DELTA,
                    patience=PATIENCE,
                    verbose=1,
                    restore_best_weights=True,
                )

            with st.spinner('Training...', show_time=True):
                history = MODEL.fit(
                    X_TRAIN,
                    Y_TRAIN,
                    batch_size=BATCH_SIZE,
                    shuffle=True,
                    epochs=EPOCHS,
                    validation_split=VALIDATION_SPLIT,
                    verbose=1,
                    callbacks=[early_stop] if EARLY_STOP else None,
                )

                MODEL.save('flavorsense_model.keras')
            return history
        # Initialization of parameters for model training
        if st.session_state.get('model_trained') != True:
            # These parameters are displayed only for the initial training of the model
            st.session_state.batch_size = st.number_input(
                'Choose a batch size',
                min_value=16,
                max_value=512,
                value=32,
                placeholder='Type a number between 16 and 512',
            )
            st.session_state.validation_split = st.number_input(
                'Choose a validation split',
                min_value=0.1,
                max_value=0.4,
                value=0.2,
                step=0.1,
                placeholder='Type a number between 0.1 and 0.4',
            )

        st.session_state.epochs = st.number_input(
            'Choose a number of epochs',
            min_value=10,
            max_value=500,
            value=10,
            placeholder='Type a number between 10 and 500',
        )

        # Setting an early stop
        st.checkbox('Early stop', key='early_stop')
        if st.session_state.early_stop:
            st.session_state.patience = st.number_input(
                'Choose a number of patience epochs',
                min_value=1,
                max_value=st.session_state.epochs,
                value=1,
                placeholder=f'Type a number between 1 and {st.session_state.epochs}',
            )
            st.session_state.min_delta = st.number_input(
                'Choose a min_delta',
                min_value=0.001,
                max_value=1.0,
                value=0.001,
                step=0.001,
                format='%0.3f',
                placeholder='Type a number between 0.001 and 1',
            )
        else:
            st.session_state.patience = None
            st.session_state.min_delta = None

        # This button tells the rest of the program that the model has been trained
        if st.sidebar.button('Train model', on_click=model_trained, args=[True]):
            st.session_state.history = train_model(
                MODEL=model,
                X_TRAIN=X_train,
                Y_TRAIN=y_train,
                BATCH_SIZE=st.session_state.batch_size,
                EPOCHS=st.session_state.epochs,
                VALIDATION_SPLIT=st.session_state.validation_split,
                EARLY_STOP=st.session_state.early_stop,
                PATIENCE=st.session_state.patience,
                MIN_DELTA=st.session_state.min_delta,
            )
            st.sidebar.success('Training complited', icon='✅')
            st.session_state.model_saved = True

        # This button deletes the trained model and tells the rest of the program that the model has not been trained
        if st.sidebar.button('Clear model', on_click=model_trained, args=[False]):
            clear_session(free_memory=True)
            del st.session_state.history
            if os.path.exists('flavorsense_model.keras'):
                os.remove('flavorsense_model.keras')
            st.session_state.model_saved = False

        if 'history' in st.session_state:
            # Showing history
            st.subheader('Showing history')
            # Function to display the history
            def show_history_plotly(history, metrics='loss', title=''):
                # Creating a DataFrame from the learning history
                if metrics == 'accuracy':
                    metrics_name = 'Точность'
                else:
                    metrics_name = 'Ошибка'

                data = {
                    'Эпоха': list(range(len(history.history[metrics]))),
                    f'{metrics_name} на обучающей выборке': history.history[metrics],
                    f'{metrics_name} на проверочной выборке': history.history['val_' + metrics],
                }
                df = pd.DataFrame(data)

                fig = px.line(
                    df,
                    x='Эпоха',
                    y=[
                        f'{metrics_name} на обучающей выборке',
                        f'{metrics_name} на проверочной выборке',
                    ],
                    title=f'{metrics_name} модели: {title}',
                    labels={'value': metrics_name, 'variable': 'Тип выборки'},
                )
                st.plotly_chart(fig, use_container_width=False, theme=None)
            # Function to display accuracy
            def show_accuracy(MODEL, X, Y):
                test_loss, test_accuracy = MODEL.evaluate(X, Y)
                st.write(f'Test accuracy: {test_accuracy*100:.2f} %')
            # Function for displaying the confusion matrix
            def create_confusion_matrix(confusion_matrix, labels):
                '''
                Visualizes the error matrix using Plotly.

                :param confusion_matrix: A 2D array or DataFrame representing an error matrix.
                :param labels: A list of class labels.
                '''
                # If the error matrix is represented as an array, convert it to a DataFrame
                if isinstance(confusion_matrix, np.ndarray):
                    confusion_matrix = (
                        pd.DataFrame(confusion_matrix, index=labels, columns=labels) * 100
                    )

                fig = px.imshow(
                    confusion_matrix,
                    labels=dict(x='Predicted', y='Actual', color='Count'),
                    x=labels,
                    y=labels,
                    text_auto='.2f',
                    aspect='equal',
                    width=500,
                )

                fig.update_layout(
                    title='Confusion Matrix',
                    title_x=0.5,
                    xaxis_title='Predicted Label',
                    yaxis_title='True Label',
                )

                st.plotly_chart(fig)

            show_history_plotly(st.session_state.history, metrics='loss', title='Плотная модель')
            show_history_plotly(st.session_state.history, metrics='accuracy', title='Плотная модель')

            # Evaluating model
            st.subheader('Evaluating model')
            show_accuracy(model, X_test, y_test)

            # Creating confusion matrix
            if st.session_state.target_encoder_name == 'Label Encoder':
                y_pred = model.predict(X_test).argmax(axis=1)
                conf_matrix = confusion_matrix(y_test, y_pred, normalize='true')
            else: 
                y_pred = model.predict(X_test)
                conf_matrix = confusion_matrix(
                    y_test.argmax(axis=1), 
                    y_pred.argmax(axis=1), 
                    normalize="true"
                )

            create_confusion_matrix(conf_matrix, st.session_state.labels)
    else:
        st.write('Please preprocess data first')

elif app_mode == 'Prediction':
    if st.session_state.get('model_saved'):
        model = load_model('flavorsense_model.keras')

        # Prediction on User data
        st.header('Prediction on User data')
        uploaded_file = st.file_uploader("Choose a file")
        if uploaded_file is not None:
            st.session_state.user_data = pd.read_csv(uploaded_file)

        with st.form("Select your data"):
            user_form = pd.DataFrame()
            age = st.number_input(
                'Choose your age',
                min_value=0,
                max_value=100,
                value=30,
                placeholder='Type a number between 0 and 100',
            )
            user_form['age'] = [age]

            user_form['sleep_cycle'] = st.selectbox(
                'Choose your sleep cycle:',
                ('Early Bird', 'Irregular', 'Night Owl'),
                index=1,
            )
            user_form['exercise_habits'] = st.selectbox(
                'Choose your exercise habits:',
                ('Light', 'Moderate', 'Heavy'),
                index=1,
            )
            user_form['climate_zone'] = st.selectbox(
                'Choose your climate_zone:',
                ('Cold', 'Temperate', 'Tropical', 'Dry'),
                index=1,
            )
            user_form['historical_cuisine_exposure'] = st.selectbox(
                'Choose your cuisine history:',
                ('Mediterranean', 'Mixed', 'Asian'),
                index=0,
            )
            if st.form_submit_button("Submit"):
                st.session_state.user_data = user_form

        if 'user_data' in st.session_state:
            st.write('You data is:')
            st.session_state.user_data
            # Scaling of numerical features
            num_user = scaler.transform(st.session_state.user_data[numeric_features])
            # Encoding of categorical features
            cat_user = encoder.transform(st.session_state.user_data[categorical_features])
            # Combining numerical and categorical features
            X_user = np.hstack((cat_user, num_user))
            # Encoding of target feature
            if st.session_state.target_encoder_name == "Label Encoder":
                y_pred_user = model.predict(X_user).argmax(axis=1)
                y_user = target_encoder.inverse_transform(y_pred_user)[0]
            else:
                y_pred_user = model.predict(X_user)
                y_user = target_encoder.inverse_transform(y_pred_user)[0][0]
            # Display the prediction
            st.write(f"Your preferet taste is {y_user}!")
        else:
            st.write('Please upload your user data or build it in the section above')
    else:
        st.write('Please train the model first')
