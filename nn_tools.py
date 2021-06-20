import tensorflow as tf
import pandas as pd
import numpy as np
from numpy import load, save
import joblib

from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from category_encoders import OneHotEncoder
from Transformers import QuantileTransformerDf, IterativeImputerDf, RareLabelNanEncoder

#############################         METRICS        ######################################
def mre(y_true, y_pred):
    # implementation of mean relative error in percents
    # Convert nparrays  to tensors
    y_true = tf.convert_to_tensor(y_true, dtype=tf.float32)
    y_pred = tf.convert_to_tensor(y_pred, dtype=tf.float32)
    MRE = tf.math.abs((y_pred - y_true) / y_true)
    MRE = tf.reduce_mean(MRE)
    return 100 * MRE


def mremax(y_true, y_pred):
    # implementation of maximum relative error in percents
    # Convert nparrays  to tensors
    y_true = tf.convert_to_tensor(y_true, dtype=tf.float32)
    y_pred = tf.convert_to_tensor(y_pred, dtype=tf.float32)
    MRE = tf.math.abs((y_pred - y_true) / y_true)
    MRE = tf.math.reduce_max(MRE)
    return 100 * MRE


def rmsle(y_true, y_pred):
    # Implementation of rmsle error
    # Convert nparrays  to tensors
    y_true = tf.convert_to_tensor(y_true, dtype=tf.float32)
    y_pred = tf.convert_to_tensor(y_pred, dtype=tf.float32)

    # Clip values to prevent log from values below 0
    y_true = tf.clip_by_value(y_true, clip_value_min=0, clip_value_max=np.inf)
    y_pred = tf.clip_by_value(y_pred, clip_value_min=0, clip_value_max=np.inf)
    return tf.math.sqrt(tf.reduce_mean((tf.math.log1p(y_pred) - tf.math.log1p(y_true)) ** 2))


def rmslemax(y_true, y_pred):
    # Convert nparrays  to tensors
    y_true = tf.convert_to_tensor(y_true, dtype=tf.float32)
    y_pred = tf.convert_to_tensor(y_pred, dtype=tf.float32)

    # Clip values to prevent log from values below 0
    y_true = tf.clip_by_value(y_true, clip_value_min=0, clip_value_max=np.inf)
    y_pred = tf.clip_by_value(y_pred, clip_value_min=0, clip_value_max=np.inf)
    return tf.math.sqrt(tf.math.reduce_max((tf.math.log1p(y_pred) - tf.math.log1p(y_true)) ** 2))


#############################         NEURAL NET MODELS        ######################################
def build_dense3L_model(config, X_train):
    # Implementation of dense 2 layer network with config keys:
    """
    config = {
        "lr": 0.01,
            # Layer 1 params
        "hidden1": 180,
        "activation1": "selu",
        "dropout1": 0.2,
            # Layer 2 params
        "hidden2": 150,
        "dropout2": 0.2,
        "activation2": "selu",
            # Layer 3 params
        "hidden3": 150,
        "dropout3": 0.2,
        "activation3": "selu",
        "activation_output": "selu"}
    """

    inputs = tf.keras.layers.Input(shape=(X_train.shape[1]))
    # x = tf.keras.layers.Flatten()(inputs)
    x = tf.keras.layers.LayerNormalization()(inputs)
    # layer 1
    x = tf.keras.layers.Dense(units=config["hidden1"], kernel_initializer='glorot_normal',
                              activation=config["activation1"])(x)
    x = tf.keras.layers.LayerNormalization()(x)
    x = tf.keras.layers.Dropout(config["dropout1"])(x)
    # layer 2
    x = tf.keras.layers.Dense(units=config["hidden2"], kernel_initializer='glorot_normal',
                              activation=config["activation2"])(x)
    x = tf.keras.layers.LayerNormalization()(x)
    x = tf.keras.layers.Dropout(config["dropout2"])(x)
    # layer 3
    x = tf.keras.layers.Dense(units=config["hidden3"], kernel_initializer='glorot_normal',
                              activation=config["activation3"])(x)
    x = tf.keras.layers.LayerNormalization()(x)
    x = tf.keras.layers.Dropout(config["dropout3"])(x)

    # Output layer
    outputs = tf.keras.layers.Dense(units=1, activation=config["activation_output"])(x)
    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="boston_model")

    model.compile(
        loss=rmsle,
        optimizer=tf.keras.optimizers.Adam(learning_rate=config["lr"]),
        metrics=[rmsle])
    return model


def build_dense2L_model(config, X_train):
    # Implementation of dense 2 layer network with config keys:
    """
    config = {
        "lr": 0.01,
        # Layer 1 params
        "hidden1": 180,
        "activation1": "selu",
        "dropout1": 0.2,
        # Layer 2 params
        "hidden2": 150,
        "dropout2": 0.2,
        "activation2": "selu",
        "activation_output": "selu"}
    """
    inputs = tf.keras.layers.Input(shape=(X_train.shape[1]))
    # x = tf.keras.layers.Flatten()(inputs)
    x = tf.keras.layers.LayerNormalization()(inputs)
    # layer 1
    x = tf.keras.layers.Dense(units=config["hidden1"], kernel_initializer='glorot_normal',
                              activation=config["activation1"])(x)
    x = tf.keras.layers.LayerNormalization()(x)
    x = tf.keras.layers.Dropout(config["dropout1"])(x)
    # layer 2
    x = tf.keras.layers.Dense(units=config["hidden2"], kernel_initializer='glorot_normal',
                              activation=config["activation2"])(x)
    x = tf.keras.layers.LayerNormalization()(x)
    x = tf.keras.layers.Dropout(config["dropout2"])(x)

    # Output layer
    outputs = tf.keras.layers.Dense(units=1, activation=config["activation_output"])(x)
    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="boston_model")

    model.compile(
        loss=rmsle,
        optimizer=tf.keras.optimizers.Adam(learning_rate=config["lr"]),
        metrics=[rmsle])
    return model


def build_conv3L_model(config, X_train):
    # Implementation of dense 2 layer network with config keys:
    """
    config = {
        "lr": 0.01,
            # Layer 1 params
        "filter1": 32,
        "kernel1":5,
        "activation_c1": "elu",
        "dropout_c1": 0.4, #0.08
            # Layer 2 params
        "filter2": 32,
        "kernel2": 5,
        "activation_c2": "elu",
            # Layer 3 params
        "filter3": 32,
        "kernel3": 5,
        "activation_c3": "elu",
            #output layers
        "hidden_conv":32
        "activation_hidden_conv":"elu"
        "activation_output": "linear"}
    """

    inputs = tf.keras.layers.Input(shape=(X_train.shape[1]),dtype=tf.bfloat16)
    x = tf.keras.layers.Reshape(target_shape=(-1, X_train.shape[1], 1))(inputs)
    # Layer 1
    x = tf.keras.layers.Conv1D(filters=config['filter1'],
                               kernel_size=config['kernel1'],
                               activation=config['activation_c1'],
                               kernel_initializer='glorot_normal',
                               name='Conv1D_1')(x)
    x = tf.keras.layers.Dropout(config['dropout_c1'])(x)
    # Layer 2
    x = tf.keras.layers.Conv1D(filters=config['filter2'],
                               kernel_size=config['kernel2'],
                               activation=config['activation_c2'],
                               kernel_initializer='lecun_normal',
                               name='Conv1D_2')(x)
    x = tf.keras.layers.Conv1D(filters=config['filter3'],
                               kernel_size=config['kernel3'],
                               activation=config['activation_c3'],
                               kernel_initializer='lecun_normal',
                               name='Conv1D_3')(x)
    x = tf.keras.layers.Reshape((np.shape(x)[2], np.shape(x)[3]))(x)

    x = tf.keras.layers.MaxPool1D(pool_size=2, name='MaxPooling1D')(x)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dropout(config['dropout_c3'])(x)
    x = tf.keras.layers.Dense(units=config['hidden_conv'],
                              activation=config['activation_hidden_conv'],
                              kernel_initializer='lecun_normal',
                              name='Dense_Conv1D')(x)

    outputs = tf.keras.layers.Dense(units=1,
                                    activation=config['activation_output'],
                                    kernel_initializer='lecun_normal',
                                    name='Dense_Conv1D_out')(x)
    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="boston_model")

    model.compile(
        loss=rmsle,
        optimizer=tf.keras.optimizers.Adam(learning_rate=config["lr"]),
        metrics=[rmsle])
    return model
def build_mixed_model(config, X_train):
    # Implementation of dense 2 layer network with config keys:
    """
    config = {
        "lr": 0.01,
        #CNN LAYERS
            # Layer 1 params
        "filter1": 32,
        "kernel1":5,
        "activation_c1": "elu",
        "dropout_c1": 0.4, #0.08
            # Layer 2 params
        "filter2": 32,
        "kernel2": 5,
        "activation_c2": "elu",
            # Layer 3 params
        "filter3": 32,
        "kernel3": 5,
        "activation_c3": "elu",
            #output layers
        "hidden_conv":32
        "activation_hidden_conv":"elu"
        #DENSE LAYERS
                    # Layer 1 params
        "hidden1": 180,
        "activation1": "selu",
        "dropout1": 0.2,
            # Layer 2 params
        "hidden2": 150,
        "dropout2": 0.2,
        "activation2": "selu",
            # Layer 3 params
        "hidden3": 150,
        "dropout3": 0.2,
        "activation3": "selu"


        "activation_output": "linear"}
    """

    inputs = tf.keras.layers.Input(shape=(X_train.shape[1]))
    input_conv = tf.keras.layers.Reshape(target_shape=(-1, X_train.shape[1], 1))(inputs)

    #CONVOLUTION PART
    # Layer 1
    x = tf.keras.layers.Conv1D(filters=config['filter1'],
                               kernel_size=config['kernel1'],
                               activation=config['activation_c1'],
                               kernel_initializer='glorot_normal',
                               name='Conv1D_1')(input_conv)
    x = tf.keras.layers.Dropout(config['dropout_c1'])(x)
    # Layer 2
    x = tf.keras.layers.Conv1D(filters=config['filter2'],
                               kernel_size=config['kernel2'],
                               activation=config['activation_c2'],
                               kernel_initializer='lecun_normal',
                               name='Conv1D_2')(x)
    x = tf.keras.layers.Conv1D(filters=config['filter3'],
                               kernel_size=config['kernel3'],
                               activation=config['activation_c3'],
                               kernel_initializer='lecun_normal',
                               name='Conv1D_3')(x)
    x = tf.keras.layers.Reshape((np.shape(x)[2], np.shape(x)[3]))(x)

    x = tf.keras.layers.MaxPool1D(pool_size=2, name='MaxPooling1D')(x)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dropout(config['dropout_c3'])(x)
    output_conv = tf.keras.layers.Dense(units=config['hidden_conv'],
                              activation=config['activation_hidden_conv'],
                              kernel_initializer='lecun_normal',
                              name='Dense_Conv1D')(x)

    #DENSE PART
    x = tf.keras.layers.LayerNormalization()(inputs)
    # layer 1
    x = tf.keras.layers.Dense(units=config["hidden1"], kernel_initializer='glorot_normal',
                              activation=config["activation1"],
                              name='Dense_1')(x)
    x = tf.keras.layers.LayerNormalization()(x)
    x = tf.keras.layers.Dropout(config["dropout1"])(x)
    # layer 2
    x = tf.keras.layers.Dense(units=config["hidden2"], kernel_initializer='glorot_normal',
                              activation=config["activation2"],
                              name='Dense_2')(x)
    x = tf.keras.layers.LayerNormalization()(x)
    x = tf.keras.layers.Dropout(config["dropout2"])(x)
    # layer 3
    x = tf.keras.layers.Dense(units=config["hidden3"], kernel_initializer='glorot_normal',
                              activation=config["activation3"],
                              name='Dense_3')(x)
    x = tf.keras.layers.LayerNormalization()(x)
    output_dense = tf.keras.layers.Dropout(config["dropout3"])(x)

    #Merge layers
    concatted = tf.keras.layers.Concatenate()([output_conv, output_dense])
    output = tf.keras.layers.Dense(units=1,
                                    activation=config['activation_output'],
                                    kernel_initializer='lecun_normal',
                                    name='Dense_Conv1D_out')(concatted)

    model = tf.keras.Model(inputs=inputs, outputs=output, name="boston_model")

    model.compile(
        loss=rmsle,
        optimizer=tf.keras.optimizers.Adam(learning_rate=config["lr"]),
        metrics=[rmsle])
    return model

def build_encoder_model(config, X_train):
    # Implementation of dense 2 layer network with config keys:
    """
    config = {
        "lr": 0.01,
        #CNN LAYERS
            # Layer 1 params
        "filter1": 32,
        "kernel1":5,
        "activation_c1": "elu",
        "dropout_c1": 0.4, #0.08
            # Layer 2 params
        "filter2": 32,
        "kernel2": 5,
        "activation_c2": "elu",
            # Layer 3 params
        "filter3": 32,
        "kernel3": 5,
        "activation_c3": "elu",
            #output layers
        "hidden_conv":32
        "activation_hidden_conv":"elu"
        #DENSE LAYERS
                    # Layer 1 params
        "hidden1": 180,
        "activation1": "selu",
        "dropout1": 0.2,
            # Layer 2 params
        "hidden2": 150,
        "dropout2": 0.2,
        "activation2": "selu",
            # Layer 3 params
        "hidden3": 150,
        "dropout3": 0.2,
        "activation3": "selu"


        "activation_output": "linear"}
    """

    inputs = tf.keras.layers.Input(shape=(X_train.shape[1]))

    x = tf.keras.layers.Dense(units=100, kernel_initializer='glorot_normal',
                              activation='elu',
                              name='Enc_Layer_1')(inputs)
    enc_output= tf.keras.layers.Dense(units=50, kernel_initializer='glorot_normal',
                              activation='sigmoid',
                              name='Enc_Layer_2')(x)


    input_conv = tf.keras.layers.Reshape(target_shape=(-1, 50, 1))(enc_output)

    #CONVOLUTION PART
    # Layer 1
    x = tf.keras.layers.Conv1D(filters=config['filter1'],
                               kernel_size=config['kernel1'],
                               activation=config['activation_c1'],
                               kernel_initializer='glorot_normal',
                               name='Conv1D_1')(input_conv)
    x = tf.keras.layers.Dropout(config['dropout_c1'])(x)
    # Layer 2
    x = tf.keras.layers.Conv1D(filters=config['filter2'],
                               kernel_size=config['kernel2'],
                               activation=config['activation_c2'],
                               kernel_initializer='lecun_normal',
                               name='Conv1D_2')(x)
    x = tf.keras.layers.Conv1D(filters=config['filter3'],
                               kernel_size=config['kernel3'],
                               activation=config['activation_c3'],
                               kernel_initializer='lecun_normal',
                               name='Conv1D_3')(x)
    x = tf.keras.layers.Reshape((np.shape(x)[2], np.shape(x)[3]))(x)

    x = tf.keras.layers.MaxPool1D(pool_size=2, name='MaxPooling1D')(x)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dropout(config['dropout_c3'])(x)
    output_conv = tf.keras.layers.Dense(units=config['hidden_conv'],
                              activation=config['activation_hidden_conv'],
                              kernel_initializer='lecun_normal',
                              name='Dense_Conv1D')(x)

    #DENSE PART
    x = tf.keras.layers.LayerNormalization()(enc_output)
    # layer 1
    x = tf.keras.layers.Dense(units=config["hidden1"], kernel_initializer='glorot_normal',
                              activation=config["activation1"],
                              name='Dense_1')(x)
    x = tf.keras.layers.LayerNormalization()(x)
    x = tf.keras.layers.Dropout(config["dropout1"])(x)
    # layer 2
    x = tf.keras.layers.Dense(units=config["hidden2"], kernel_initializer='glorot_normal',
                              activation=config["activation2"],
                              name='Dense_2')(x)
    x = tf.keras.layers.LayerNormalization()(x)
    x = tf.keras.layers.Dropout(config["dropout2"])(x)
    # layer 3
    x = tf.keras.layers.Dense(units=config["hidden3"], kernel_initializer='glorot_normal',
                              activation=config["activation3"],
                              name='Dense_3')(x)
    x = tf.keras.layers.LayerNormalization()(x)
    output_dense = tf.keras.layers.Dropout(config["dropout3"])(x)

    #Merge layers
    concatted = tf.keras.layers.Concatenate()([output_conv, output_dense])
    output = tf.keras.layers.Dense(units=1,
                                    activation=config['activation_output'],
                                    kernel_initializer='lecun_normal',
                                    name='Dense_Conv1D_out')(concatted)

    model = tf.keras.Model(inputs=inputs, outputs=output, name="boston_model")

    model.compile(
        loss=tf.keras.losses.KLDivergence, #tf.keras.losses.KLDivergence, rmsle
        optimizer=tf.keras.optimizers.Adam(learning_rate=config["lr"]),
        metrics=[rmsle])
    return model



def make_preprocessing(config):
    train_enc_file = f'X_train_enc_rare_tol_{config["rare_tol"]}_n_categories_{config["n_categories"]}_max_iter_{config["max_iter"]}_iter_tol_{config["iter_tol"]}.npy'
    train_enc_path = '/home/peterpirog/PycharmProjects/BostonHousesTune/data/encoded/' + train_enc_file
    y_train_path = '/home/peterpirog/PycharmProjects/BostonHousesTune/data/encoded/y_train.npy'

    pipeline_file=f'pipeline_{config["rare_tol"]}_n_categories_{config["n_categories"]}_max_iter_{config["max_iter"]}_iter_tol_{config["iter_tol"]}.pkl'
    pipeline_path='/home/peterpirog/PycharmProjects/BostonHousesTune/data/'+pipeline_file

    try:
        joblib.load(pipeline_path, mmap_mode=None)
        X_train_encoded = load(file=train_enc_path)
        Y_train = load(file=y_train_path)

    except:
        df_train = pd.read_csv('/home/peterpirog/PycharmProjects/BostonHousesTune/data/train.csv')

        # csv preprcessing
        df_train['MSSubClass'] = df_train['MSSubClass'].astype(dtype='category')  # convert feature to categorical
        X_train = df_train.drop(['Id', 'SalePrice'], axis=1)
        Y_train = df_train['SalePrice'].astype(dtype=np.float32)

        # PREPROCESSING
        # STEP 1 -  categorical features rare labels encoding
        rle = RareLabelNanEncoder(categories=None, tol=config['rare_tol'],
                                  minimum_occurrences=None,
                                  n_categories=config['n_categories'],
                                  max_n_categories=None,
                                  replace_with='Rare',
                                  impute_missing_label=False,
                                  additional_categories_list=None)

        # STEP 2 - categorical features one hot encoding
        # https://github.com/scikit-learn-contrib/category_encoders/blob/master/category_encoders/one_hot.py
        ohe = OneHotEncoder(verbose=0, cols=None, drop_invariant=False, return_df=True,
                            handle_missing='return_nan',
                            # options are 'error', 'return_nan', 'value', and 'indicator'.
                            handle_unknown='return_nan',
                            # options are 'error', 'return_nan', 'value', and 'indicator'
                            use_cat_names=False)

        # STEP 3 - numerical values quantile transformation with skewness removing
        q_trans = QuantileTransformerDf(n_quantiles=1000, output_distribution='uniform',
                                        ignore_implicit_zeros=False,
                                        subsample=1e5, random_state=42, copy=True, dataframe_as_output=True,
                                        dtype=np.float32)

        # STEP 4 - missing values multivariate imputation
        imp = IterativeImputerDf(min_value=0,  # values from 0 to 1 for categorical for numeric
                                 max_value=1,
                                 random_state=42,
                                 initial_strategy='median',
                                 max_iter=config['max_iter'],
                                 tol=config['iter_tol'],
                                 verbose=3, dataframe_as_output=False)
        # Step 5 PCA
        pca = PCA(n_components=config['n_components'], svd_solver='full')

        # STEP 6 MAKE PIPELINE AND TRAIN IT
        pipeline = Pipeline([
            ('rare_lab', rle),
            ('one_hot', ohe),
            ('q_trans', q_trans),
            ('imputer', imp),
            ('pca', pca)
        ])

        # Pipeline training
        pipeline.fit(X_train)
        X_train_encoded = pipeline.transform(X_train).astype(dtype=np.float32)
        # save X_train_encoded array
        save(file=train_enc_path, arr=X_train_encoded)
        save(file=y_train_path, arr=Y_train)

        #save trained pipeline
        joblib.dump(pipeline, pipeline_path)

    # STEP 7 SPLITTING DATA FOR KERAS
    X_train, X_test, y_train, y_test = train_test_split(X_train_encoded, Y_train,
                                                        shuffle=True,
                                                        test_size=0.2,
                                                        random_state=42)
    return X_train, X_test, y_train, y_test

