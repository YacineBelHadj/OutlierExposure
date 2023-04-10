

# %%
import matplotlib.pyplot as plt
import numpy as np
from OutlierExposure.utils import add_event

def select_row(df, column:str='sensor', value:str='ACC2_Z'):
    return df[df[column] == value]

def plot_control_chart(df, sensor_name='ACC2_Z', rolling_prod=0, ax=None):
    # Select the relevant rows from the dataframe
    df_selected = df.loc[df['sensor'] == sensor_name].copy()
    start_val, end_val = df.index[df['validation'].diff()==True]
    # Create the figure and axis if not provided
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 6))

    # Calculate the y values for the control chart
    if rolling_prod > 0:
        df_selected['confidence'] = df_selected['confidence'].rolling(rolling_prod).apply(np.prod, raw=True)
    else:
        df_selected['confidence'] = df_selected['confidence']

    # Plot the control chart
    ax.plot(df_selected.index, df_selected['confidence'].values, marker='.', linestyle='')

    y_lowbound, y_upbound= ax.get_ylim()
    middle = (y_lowbound + y_upbound) / 2
    range = (y_upbound - y_lowbound) / 2
    text_y = middle - range * 0.1

    ax.axvline(x=start_val, color='green', linestyle='--')
    ax.text(start_val, text_y, 'Test data', rotation=90, verticalalignment='center')
    # Calculate the control limits and plot them
    confidence_train = df_selected['confidence'][df_selected['train']].dropna()
    mean = np.mean(confidence_train)
    std = np.std(confidence_train)
    lcl = mean - 2 * std
    ax.axhline(y=lcl, color='red', linestyle='--')

    # Highlight any points that fall below the lower control limit
    alert_df = df_selected[df_selected['confidence'] < lcl]
    ax.plot(alert_df.index, alert_df['confidence'], marker='.', linestyle='', color='red')
    # Add any relevant events to the plot
    add_event(ax)

    # Set the axis labels and title
    ax.set_ylabel('Confidence')
    ax.set_xlabel('Index')
    ax.set_title(sensor_name)

    return fig, ax