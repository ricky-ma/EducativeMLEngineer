import pandas as pd
import matplotlib.pyplot as plt


# Read the CSV data files into DataFrames
def read_dataframes():
    train_df = pd.read_csv('weekly_sales.csv')
    features_df = pd.read_csv('features.csv')
    stores_df = pd.read_csv('stores.csv')
    return train_df, features_df, stores_df


# Fill in missing data w/ nearby data
def impute_data(merged_features, na_indexes_cpi, na_indexes_une):
    for i in na_indexes_cpi:
        merged_features.at[i, 'CPI'] = merged_features.at[i - 1, 'CPI']
    for i in na_indexes_une:
        merged_features.at[i, 'Unemployment'] = merged_features.at[i - 1, 'Unemployment']
    return merged_features


# Data cleaning pipeline
def clean_data():
    train_df, features_df, stores_df = read_dataframes()
    merged_features = features_df.merge(stores_df, on='Store')

    # Get rows with missing values
    na_values = pd.isna(merged_features)
    na_cpi_int = na_values['CPI'].astype(int)
    na_indexes_cpi = na_cpi_int.to_numpy().nonzero()[0]
    na_une_int = na_values['Unemployment'].astype(int)
    na_indexes_une = na_une_int.to_numpy().nonzero()[0]

    # Form main dataset
    merged_features = impute_data(merged_features, na_indexes_cpi, na_indexes_une)
    features = ['Store', 'Date', 'IsHoliday']
    final_dataset = train_df.merge(merged_features, on=features)
    final_dataset = final_dataset.drop(columns=['Date'])

    # Get class labels
    type_labels = final_dataset['Type'].unique()
    dept_labels = final_dataset['Dept'].unique()
    holi_labels = final_dataset['IsHoliday'] = final_dataset['IsHoliday'].astype(int)

    return final_dataset, type_labels, dept_labels, holi_labels


def plot_temp_vs_weekly_sales(final_dataset):
    plot_df = final_dataset[['Weekly_Sales', 'Temperature']]
    rounded_temp = plot_df['Temperature'].round(0)  # nearest integer
    plot_df = plot_df.groupby(rounded_temp).mean()
    plot_df.plot.scatter(x='Temperature', y='Weekly_Sales')
    plt.title('Temperature vs. Weekly Sales')
    plt.xlabel('Temperature (Fahrenheit)')
    plt.ylabel('Avg Weekly Sales (Dollars)')
    plt.show()


def plot_store_type_vs_weekly_sales(final_dataset):
    plot_df = final_dataset[['Weekly_Sales', 'Type']]
    plot_df = plot_df.groupby('Type').mean()
    plot_df.plot.bar()
    plt.title('Store Type vs. Weekly Sales')
    plt.xlabel('Type')
    plt.ylabel('Avg Weekly Sales (Dollars)')
    plt.show()
