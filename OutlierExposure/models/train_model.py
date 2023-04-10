#%%
from OutlierExposure.data import formatting as fmt
from OutlierExposure.classification.dense_nn import DenseSignalClassifier
import tensorflow as tf
import numpy as np
import pandas as pd
import mlflow
import mlflow.tensorflow
from sklearn.metrics import confusion_matrix, f1_score
from OutlierExposure.utils import add_event
from OutlierExposure.visualization.visualize import plot_control_chart, select_row
import sys
import matplotlib.pyplot as plt
import seaborn as sns
EPS = sys.float_info.epsilon

df,freq = fmt.load_psd_from_hdf5()
df=fmt.remove_row(df,'sensor',['ACC1_X','ACC1_Y'])
df = df[df.index>='2022-04-02']
df['psd']=df['psd'].apply(lambda x:np.log(x+EPS))
df['psd']=fmt.normalize(df)
encoded,encoder=fmt.one_hot_encode(df['sensor'])
df=fmt.train_val_test_split(df)

sensors_name = df['sensor'].unique()

x = fmt.get_array(df, column='psd') 
y = encoded.toarray()

train_data = (df['train']==True).values
validation_data = (df['validation']==True).values
x_train, y_train = x[train_data],  y[train_data]
x_test, y_test = x[~train_data], y[~train_data]



 

#%%
loss_function = 'categorical_crossentropy'
layers =[32,16]
optimizer = 'adam'
activation = 'ReLU'
batch_norm = True
dropout_rate = 0
model_name = 'DenseSignalClassifier'
model = eval(model_name)(num_class=y_train.shape[1],
                            inputDim=x_train.shape[1],
                            dense_layers=layers,
                            dropout_rate=dropout_rate,
                            batch_norm=batch_norm,
                            activation=activation)


mlflow.set_experiment("your_experiment_name")

with mlflow.start_run():
    model = model.build_model()

    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath="best_model.h5",
        save_best_only=True,
        monitor='val_loss',
        mode='min',
        verbose=1)

    early_stopping_callback = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=10,
        mode='min',
        restore_best_weights=True)

    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', 
                                                        factor=0.1, 
                                                        patience=5, 
                                                        min_lr=1e-6)

    model.fit(x_train,y_train,epochs=100,validation_split=0.2,
              callbacks=[model_checkpoint_callback,early_stopping_callback,reduce_lr])
    output = model.predict(x)
    true_class = np.argmax(y,axis=1)
    true_class_val = np.argmax(y_train,axis=1)
    pred_class_val = np.argmax(output[train_data],axis=1)
    x_val = x[validation_data]
    y_val = y[validation_data]
    confidence = output[np.arange(len(true_class)), true_class]
    df['confidence'] = confidence

    confusion = confusion_matrix(true_class_val, pred_class_val)
    sns.heatmap(pd.DataFrame(confusion)).get_figure().savefig('docs/'+'confusion.png')
    mlflow.log_artifact('docs/'+'confusion.png', 'confusion_matrix')

    val_loss, val_accuracy = model.evaluate(x_val, y_val, verbose=0)
    mlflow.log_metric("validation_loss", val_loss)
    mlflow.log_metric("validation_accuracy", val_accuracy)

    f1 = f1_score(true_class_val, pred_class_val, average='weighted')
    mlflow.log_metric("f1_score", f1)

    for sensor in sensors_name:

        fig,ax =plot_control_chart(df, sensor_name=sensor, rolling_prod=0)
        fig.savefig('docs/'+sensor+'.png')
        mlflow.log_artifact('docs/'+sensor+'.png','Control chart')
        plt.close(fig)

