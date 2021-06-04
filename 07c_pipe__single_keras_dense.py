# https://www.machinecurve.com/index.php/2020/02/18/how-to-use-k-fold-cross-validation-with-keras/

import numpy as np
import tensorflow as tf  # tensorflow >= 2.5
import pandas as pd

import ray
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from category_encoders import OneHotEncoder

# Own functions and classes
from Transformers import QuantileTransformerDf, IterativeImputerDf, RareLabelNanEncoder
from own_metrics import rmsle_tf, mre_tf

if __name__ == "__main__":
    # https://github.com/tensorflow/tensorflow/issues/32159

    # print('Is cuda available for trainer:', tf.config.list_physical_devices('GPU'))

    # INITIAL CONFIGURATION
    config = {
        # preprocessing parameters
        # RARE LABEL ENCODER  https://feature-engine.readthedocs.io/en/latest/encoding/RareLabelEncoder.html
        "rare_tol": 0.05,
        # The minimum frequency a label should have to be considered frequent. Categories with frequencies lower than tol will be grouped
        "n_categories": 10,  # he minimum number of categories a variable should have for the encoder to find
        # frequent labels. If the variable contains less categories, all of them will be considered frequent.

        # ITERATIVE IMPUTER https://scikit-learn.org/stable/modules/generated/sklearn.impute.IterativeImputer.html
        "max_iter": 10,
        "iter_tol": 1e-3,

        # PCA DECOMPOSITION https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html
        "n_components": 0.99,  # number of components such that the amount of variance that needs to be explained is
        # greater than the percentage specified by n_components.

        # NEURAL NET PARAMETERS
        "batch": 4,
        "lr": 0.01,
        # Layer 1 params
        "hidden1": 120,
        "activation1": "elu",
        "dropout1": 0.4,
        # Layer 2 params
        "hidden2": 97,
        "dropout2": 0.4,
        "activation2": "elu",
        "activation_output": "elu"
    }

    df_train = pd.read_csv('data/train.csv')

    # csv preprcessing
    df_train['MSSubClass'] = df_train['MSSubClass'].astype(dtype='category')  # convert fature to categorical
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
                        handle_missing='return_nan',  # options are 'error', 'return_nan', 'value', and 'indicator'.
                        handle_unknown='return_nan',  # options are 'error', 'return_nan', 'value', and 'indicator'
                        use_cat_names=False)

    # STEP 3 - numerical values quantile transformation with skewness removing
    q_trans = QuantileTransformerDf(n_quantiles=1000, output_distribution='uniform', ignore_implicit_zeros=False,
                                    subsample=1e5, random_state=42, copy=True, dataframe_as_output=True,
                                    dtype=np.float32)

    # STEP 4 - missing values multivariate imputation
    imp = IterativeImputerDf(min_value=0,  # values from 0 to 1 for categorical for numeric
                             max_value=1,
                             random_state=42,
                             initial_strategy='median',
                             max_iter=config['max_iter'],
                             tol=config['iter_tol'],
                             verbose=0, dataframe_as_output=False)
    # Step 5 PCA
    pca = PCA(n_components=config['n_components'], svd_solver='full')

    # STEP 6 MAKE PIPELINE AND TRAIN IT
    pipe = Pipeline([
        ('rare_lab', rle),
        ('one_hot', ohe),
        ('q_trans', q_trans),
        ('imputer', imp),
        ('pca', pca)
    ])

    # Pipeline training
    pipe.fit(X_train)
    X_train_encoded = pipe.transform(X_train)

    # STEP 7 SPLITTING DATA FOR KERAS
    X_train, X_test, y_train, y_test = train_test_split(X_train_encoded, Y_train,
                                                        shuffle=True,
                                                        test_size=0.2,
                                                        random_state=42)

    # STEP 8 DEFINE NEURAL NET MODEL
    epochs = 1000

    inputs = tf.keras.layers.Input(shape=(X_train.shape[1]))
    x = tf.keras.layers.Flatten()(inputs)
    x = tf.keras.layers.LayerNormalization()(x)
    # layer 1
    x = tf.keras.layers.Dense(units=config["hidden1"], kernel_initializer='glorot_normal',
                              activation=config["activation1"])(x)
    x = tf.keras.layers.BatchNormalization(x)
    x = tf.keras.layers.Dropout(config["dropout1"])(x)
    # layer 2
    x = tf.keras.layers.Dense(units=config["hidden2"], kernel_initializer='glorot_normal',
                              activation=config["activation2"])(x)
    x = tf.keras.layers.BatchNormalization(x)
    x = tf.keras.layers.Dropout(config["dropout2"])(x)
    outputs = tf.keras.layers.Dense(units=1, activation=config["activation_output"])(x)
    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="boston_model")

    model.compile(
        loss=rmsle_tf,  # mean_squared_logarithmic_error
        optimizer=tf.keras.optimizers.Adam(learning_rate=config["lr"]),
        metrics=[mre_tf])  # absolute relative error in %

    # Define callbacks
    callbacks_list = [tf.keras.callbacks.EarlyStopping(monitor='val_mre_tf',
                                                       patience=15),
                      tf.keras.callbacks.ReduceLROnPlateau(monitor='val_mre_tf',
                                                           factor=0.1,
                                                           patience=10)]  # ,
    """
                    tf.keras.callbacks.ModelCheckpoint(filepath='my_model.h5',
                                                       monitor='val_rmsle_tf',
                                                       save_best_only=True)]
    """
    # STEP 9 TRAIN NEURAL NET
    history = model.fit(
        X_train,
        y_train,
        batch_size=config["batch"],
        epochs=epochs,
        verbose=1,
        validation_data=(X_test, y_test),
        callbacks=callbacks_list)

    history_dict = history.history

    error = np.array(history.history['mre_tf'])
    loss = np.array(history.history['loss'])
    val_error = np.array(history.history['val_mre_tf'])
    val_loss = np.array(history.history['val_loss'])

    start_iter = 20
    plt.plot(loss[start_iter:], 'b', label="Błąd trenowania")
    plt.plot(val_loss[start_iter:], 'bo', label="Błąd walidacji")
    plt.xlabel("Epoki")
    plt.ylabel('Strata')
    plt.legend()
    plt.show()
