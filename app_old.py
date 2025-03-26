import streamlit as st
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import json
import os
from utilities import run_kmeans, create_views, download_data
from PIL import Image
import zipfile
from datetime import datetime, timedelta

# Data Fetch function
def fetch_data():
    # Define data folder
    folder_path = "D:/AutomaticClustering/data"
    file_list = os.listdir(folder_path)
    # Calculate the date of the end of the previous month
    current_date = datetime.now()
    last_day_of_previous_month = datetime(current_date.year, current_date.month, 1) - timedelta(days=1)
    # Format the last day of the previous month as YYYY_MM_DD
    last_day_formatted = last_day_of_previous_month.strftime('%Y_%m_%d')

    if last_day_formatted+".feather" in file_list:
        data_init = pd.read_feather("data/{}".format(last_day_formatted+".feather"))
    else:
        create_views()
        data_init = download_data()
        data_init.to_feather("data/{}".format(last_day_formatted+".feather"))
        # New data saved
    return data_init

# Download data
df_init = fetch_data()

# Preprocess data
df_init['TENURE'] = df_init['TENURE'].fillna(0)
df_init['TENURE'] = df_init['TENURE'].apply(lambda x: min(x, 30.0))
df_init['IS_INT_BNK_USER'] = np.where(df_init['IS_INT_BNK_USER'] == '1', 1, np.float32(0))
df_init['IS_MOB_BNK_USER'] = np.where(df_init['IS_INT_BNK_USER'] == '1', 1, np.float32(0))
df_init['PREF_CITY'] = df_init['PREF_CITY'].fillna("Undefined")
df_init['PAYROLL_TP'] = df_init['PAYROLL_TP'].fillna("Undefined")
df_init['SOC_SEG'] = df_init['SOC_SEG'].fillna("Undefined")
df_init['AVG_MB_IB_TRN_CNT'] = df_init['AVG_MB_IB_TRN_CNT'].fillna(0)
df_init['AVG_POS_TRN_CNT'] = df_init['AVG_POS_TRN_CNT'].fillna(0)
df_init['AVG_ATM_TRN_CNT'] = df_init['AVG_ATM_TRN_CNT'].fillna(0)
df_init['MONTHLY_INCOME'] = df_init['MONTHLY_INCOME'].fillna(0)


# Import Json data
with open("variables.json", 'r', encoding='utf-8') as json_file:
    data_variables = json.load(json_file)

global_dictionary={}

# Main function to create Streamlit app
def main():
    # Set Streamlit app configuration
    # Define page config
    st.set_page_config(
        page_title="Clustering App by Optio",
        layout="wide"
    )
    
    st.image('logo.png', width=300)

    # Add statement at the top
    st.markdown(
        """
        <h1 style='text-align: center; font-size: 50px;'>Clustering Application MVP</h1>
        <h1 style='text-align: center; font-size: 48px;'>by Optio</h1>
        """,
        unsafe_allow_html=True
    )
    # Data Filtering
    st.header("Data filtering", help="In this module you can filter initial customer base")
    st.write("\n")
    with st.form("filtering_form"):
        #
        st.markdown('<h3 style="font-size: 20px;">Choose a range of AGE</h3>', unsafe_allow_html=True)
        var_age = st.slider("", min_value=0.0, max_value=110.0, value=(18.0, 65.0), step=1.0, key ='1')
        #
        st.markdown('<h3 style="font-size: 20px;">Choose a range of TENURE</h3>', unsafe_allow_html=True)
        var_tenure = st.slider("", min_value = 0.0, max_value=30.0, value=(0.0, 30.0), step=1.0, key ='2')
        #
        st.markdown('<h3 style="font-size: 20px;">Choose a range of MONTHLY INCOME</h3>', unsafe_allow_html=True)
        var_income = st.slider("", min_value = 0.0, max_value=10000.0, value=(0.0, 5000.0), step=50.0, key ='3')
        #
        st.markdown('<h3 style="font-size: 20px;">Choose a GENDER</h3>', unsafe_allow_html=True)
        var_gender = st.multiselect("", options=['F', 'M'], default=['F', 'M'], key ='4')
        #
        st.markdown('<h3 style="font-size: 20px;">Choose a CITY</h3>', unsafe_allow_html=True)
        var_city = st.multiselect("", options=['ბორჯომი', 'ზუგდიდი', 'თბილისი', 'გორი', 'ქუთაისი', 'საჩხერე',
        'ბათუმი', 'ჭიათურა', 'ხაშური', 'ჩოხატაური', 'ხობი',
        'წყალტუბო', 'ახალციხე', 'ყვარელი', 'მარტვილი', 'რუსთავი',
        'ქობულეთი', 'ხელვაჩაური', 'ფოთი', 'გურჯაანი', 'მარნეული', 'ონი',
        'ლაგოდეხი', 'ბოლნისი', 'გარდაბანი', 'ვანი', 'ლენტეხი', 'საგარეჯო',
        'დუშეთი', 'ასპინძა', 'სამტრედია', 'ამბროლაური', 'სენაკი',
        'ლანჩხუთი', 'თიანეთი', 'წალენჯიხა', 'აბაშა', 'ოზურგეთი',
        'ნინოწმინდა', 'სიღნაღი', 'დედოფლისწყარო', 'ქარელი', 'ზესტაფონი',
        'თერჯოლა', 'მცხეთა', 'დმანისი', 'თელავი', 'ხონი', 'ახალქალაქი',
        'ბაღდათი', 'ხულო', 'კასპი', 'ცაგერი', 'ახმეტა', 'წალკა',
        'ჩხოროწყუ', 'ყაზბეგი', 'შუახევი', 'ხარაგაული', 'ტყიბული',
        'ადიგენი', 'თყიბული', 'ქვარელი', 'ცალქა', 'მართვილი', 'მესტია',
        'თეთრიწყარო', 'თეთრიცყარო', 'ქედა', 'კასფი', 'ცყალთუბო',
        'დედოფლისცყარო', 'ახალგორი', 'ცალენჯიხა', 'ქასფი', 'სენაქი',
        'სამთრედია', 'ახმეთა', 'მესთია', 'ჩოხათაური', 'ზესთაფონი',
        'ჩიათურა', 'ჩხოროცყუ', 'ნინოცმინდა', 'ასფინძა', 'ლენთეხი', 'Undefined'], default=['თბილისი'], key ='5')
        #
        st.markdown('<h3 style="font-size: 20px;">Choose a RESIDENCY</h3>', unsafe_allow_html=True)
        var_residency = st.multiselect("", options=[0, 1], default = [0, 1], key ='6')
        #
        st.markdown('<h3 style="font-size: 20px;">Choose a PAYROLL TYPE</h3>', unsafe_allow_html=True)
        var_payroll = st.multiselect("", options=['Other Payroll', 'Municipal Payroll', 'Remittance','Teacher Payroll', 'Staff', 'Undefined'], default=['Other Payroll'], key ='7')
        #
        st.markdown('<h3 style="font-size: 20px;">Choose a SOCIAL SEGMENT</h3>', unsafe_allow_html=True)
        var_social = st.multiselect("", options=['SOB', 'ELB', 'DEV', 'Undefined', 'DZA', 'UMW', 'CHLD', 'COV'], default=['SOB', 'ELB', 'DEV', 'Undefined', 'DZA', 'UMW', 'CHLD', 'COV'], key ='8')
        #
        st.markdown('<h3 style="font-size: 20px;">Choose WHETHER CLIENT IS INT BANK USER</h3>', unsafe_allow_html=True)
        var_int_bnk = st.multiselect("", options=[0.0, 1.0], default = [0.0, 1.0], key ='9')
        #
        st.markdown('<h3 style="font-size: 20px;">Choose WHETHER CLIENT IS MOB BANK USER</h3>', unsafe_allow_html=True)
        var_mob_bnk = st.multiselect("", options=[0.0, 1.0], default = [0.0, 1.0], key ='10')
        #
        st.markdown('<h3 style="font-size: 20px;">Choose a range of AVERAGE TRANSACTION COUNT IN MB&INT BANKS</h3>', unsafe_allow_html=True)
        var_mb_trn_cnt = st.slider("", min_value=0.0, max_value=1500.0, value=(0.0, 100.0), step=1.0, key ='11')
        #
        st.markdown('<h3 style="font-size: 20px;">Choose a range of AVERAGE TRANSACTION COUNT ON POS</h3>', unsafe_allow_html=True)
        var_pos_trn_cnt = st.slider("", min_value=0.0, max_value=1000.0, value=(0.0, 100.0), step=1.0, key ='12')
        #
        st.markdown('<h3 style="font-size: 20px;">Choose a range of AVERAGE TRANSACTION COUNT ON ATM</h3>', unsafe_allow_html=True)
        var_atm_trn_cnt = st.slider("", min_value=0.0, max_value=100.0, value=(0.0, 50.0), step=1.0, key ='13')

        submitted_filter = st.form_submit_button("Submit Filters")

        @st.cache_data
        def generate_df(df, var_age, var_tenure, var_income, var_gender, var_city, var_residency, var_payroll, var_social, var_int_bnk, var_mob_bnk, \
            var_mb_trn_cnt, var_pos_trn_cnt, var_atm_trn_cnt):
            #1
            df = df[df['AGE'].between(*var_age)]
            #2
            # print(1)
            # print(df.shape)
            df = df[df['TENURE'].between(*var_tenure)]
            # print(2)
            #3
            # print(df.shape)
            df = df[df['MONTHLY_INCOME'].between(*var_income)]
            # print(3)
            #4
            # print(df.shape)
            df = df[df['GENDER'].isin(var_gender)]
            # print(4)
            #5
            # print(df.shape)
            df = df[df['PREF_CITY'].isin(var_city)]
            # print(5)
            #6
            # print(df.shape)
            df = df[df['RES_FLAG'].isin(var_residency)]
            # print(6)
            #7
            # print(df.shape)
            df = df[df['PAYROLL_TP'].isin(var_payroll)]
            # print(7)
            #8
            # print(df.shape)
            df = df[df['SOC_SEG'].isin(var_social)]
            # print(8)
            #9
            # print(df.shape)
            df = df[df['IS_INT_BNK_USER'].isin(var_int_bnk)]
            # print(9)
            #10
            # print(df.shape)
            df = df[df['IS_MOB_BNK_USER'].isin(var_mob_bnk)]
            # print(10)
            #11
            # print(df.shape)
            df = df[df['AVG_MB_IB_TRN_CNT'].between(*var_mb_trn_cnt)]
            # print(11)
            #12
            # print(df.shape)
            df = df[df['AVG_POS_TRN_CNT'].between(*var_pos_trn_cnt)]
            # print(12)
            #13
            # print(df.shape)
            df = df[df['AVG_ATM_TRN_CNT'].between(*var_atm_trn_cnt)]
            # print(13)
            # print(df.shape)
            return df

        # Generate filtered dataframe
        df=generate_df(df_init, var_age, var_tenure, var_income, var_gender, var_city, var_residency, var_payroll, var_social, var_int_bnk, var_mob_bnk, \
            var_mb_trn_cnt, var_pos_trn_cnt, var_atm_trn_cnt)

        # Save filtered dataframe
        global_dictionary['dataframe'] = df 

        if submitted_filter:
            col1, col2 = st.columns(2)
            col1.metric("Total Number of Customers", "{:,}".format(df.shape[0]))
            col2.metric("Total Number of Variables", str(df.shape[1]))

            
    st.write("\n")
    st.write("\n")
    # Set app title and description
    st.header("Variable Selection", help="In this module you have to select variables that you think is appropriate for K-Means modeling")
    st.write("\n")

    # df = df_init.fillna(0)
    # Define model variables
    with st.form("variable_form"):

        # Define modeling variables
        model_variables = data_variables["model_variables"]
        # Define reporting variables
        reporting_variables = data_variables["reporting_variables"]

        # Choose model variables from the dropdown menu
        st.markdown('<h3 style="font-size: 20px;">Select modeling variables</h3>', unsafe_allow_html=True)
        selected_model_columns = st.multiselect(" ", model_variables, default= ['MONTHLY_INCOME', 'DEPO_AMOUNT', 'LOAN_AMOUNT'])
        st.write("\n")
        # Choose model variables from the dropdown menu
        st.markdown('<h3 style="font-size: 20px;">Select reporting variables</h3>', unsafe_allow_html=True)
        selected_reporting_columns = st.multiselect(" ", reporting_variables)
        st.write("\n")

        global_dictionary['selected_model_columns']=selected_model_columns
        global_dictionary['selected_reporting_columns']=selected_reporting_columns

        submitted_selection = st.form_submit_button("Submit Selection")

        if submitted_selection:
            st.success('', icon="✅")


    # Plot correlation heatmap using seaborn
    st.header("Pair-Wise Correlation", help="This module helps to opt out unnecessary modeling variables")

    with st.form("correlation_form"):
        # st.markdown('<p class="big-font">It is commonly recommended to avoid including variables in the final model that exhibit a \
        # <span style="color:red">strong positive (above 0.8)</span> or a \
        # <span style="color:blue">strong negative correlation (below -0.8)</span>\
        # as it indicates that they essentially provide similar information, and it might not be necessary \
        # to include both of them in the model. It is advisable to opt for just one variable from each correlated pair for inclusion in the model. </p>', unsafe_allow_html=True)
        st.markdown("""
        <style>
            .big-font {
                font-size: 25px;
            }
            .red-text {
                color: #e11d48;
                font-weight: bold;
                font-size: 27px;
            }
            .blue-text {
                color: #4338ca;
                font-weight: bold;
                font-size: 27px;
            }
        </style>
        <p class="big-font">It is commonly recommended to avoid including variables in the final model that exhibit a 
        <span class="red-text">strong positive (above 0.8)</span> or a 
        <span class="blue-text">strong negative correlation (below -0.8)</span> as it indicates that they essentially provide similar information, and it might not be necessary 
        to include both of them in the model. It is advisable to opt for just one variable from each correlated pair for inclusion in the model.</p>
        """, unsafe_allow_html=True)

        st.write("\n")
        st.write("\n")
        st.write("\n")

        def calculate_correlation(df):
            selected_model_columns=global_dictionary['selected_model_columns']
            corr = round(df[selected_model_columns].corr(), 2)
            # mask = np.zeros_like(corr, dtype= bool)
            # mask[np.triu_indices_from(mask)] = True
            mask = np.triu(np.ones_like(corr, dtype=bool), k=1.0)
            fig, ax = plt.subplots(figsize=(15, 15))
            cmap = sns.diverging_palette(220, 10, as_cmap=True)
            sns.heatmap(
            corr,          
            mask=mask,     
            cmap=cmap,    
            annot=True,   
            vmax=1.0,       
            vmin=-1.0,      
            center=0,      
            square=True,   
            linewidths=.5, 
            cbar_kws={"shrink": .5}
            )

            st.pyplot(fig)
        
        submitted_corr = st.form_submit_button("Calculate Correlations")

        if submitted_corr:
            df = global_dictionary['dataframe']
            calculate_correlation(df)
        
    # Choose the value of K
    st.write("\n")
    st.write("\n")
    st.header("Modeling", help="This module calculates k different models and plots Elbow Graph")

    with st.form("form_modeling"):
        st.markdown('<h3 style="font-size: 20px;">Choose the value of K</h3>', unsafe_allow_html=True)
        k = st.slider(" ", min_value = 3, max_value = 30)

        # Calculate button
        def data_transformation(df):
            # Filter the dataframe based on selected variables
            selected_model_columns = global_dictionary['selected_model_columns']
            selected_df = df[selected_model_columns]
            selected_df = selected_df.fillna(0)
            # Standardize the selected data
            scaler = StandardScaler()
            scaled_selected_data = scaler.fit_transform(selected_df[selected_model_columns])

            return scaled_selected_data

        def train_model(df, k):
            progress_bar = st.progress(0, text="Calculating K-Means model with 3 clusters")
            # Run K-means algorithm for each value of K
            elbow_data = []
            for i in range(3, k + 1):
                kmeans = KMeans(n_clusters=i, random_state=42)
                kmeans.fit(df)
                elbow_data.append({"K": i, "Inertia": kmeans.inertia_})

                # Update the progress bar
                progress = (i - 2) / (k - 2)
                progress_bar.progress(progress, text="Calculating K-Means model with {} clusters".format(i))
            progress_bar.empty()
            st.write("Calculation Finished")
            st.write("Showing Elbow Graph")

            st.write("\n")
            # Create plotly graph
            elbow_df = pd.DataFrame(elbow_data)
            fig = px.line(elbow_df, x='K', y='Inertia', markers = True)
            # Customize layout to increase graph size
            fig.update_layout(
                width=800,    # Set the width of the figure in pixels
                height=600,   # Set the height of the figure in pixels
                xaxis_title='Number of Clusters (K)',
                yaxis_title='Measure of Heterogeneity',
                xaxis=dict(showgrid=True, title_font = dict(size = 22)),  # Show vertical grid lines
                yaxis=dict(showgrid=False, title_font = dict(size = 22))  # Hide horizontal grid lines
            )

            # Update traces (optional, if needed)
            fig.update_traces(
                mode='lines+markers',
                line=dict(color='#075985', width=4),
                marker=dict(symbol='circle', size=12),
                text=elbow_df['Inertia'].round(2),
                textposition='top center'
            )
            fig.update_xaxes(tickmode='linear', tick0=0, dtick=1, tickfont=dict(size=22))

            # Show Elbow graph
            st.plotly_chart(fig, use_container_width= True)
        
        submitted_models = st.form_submit_button("Run The Experiment")

        if submitted_models:
            df = global_dictionary['dataframe']
            scaled_selected_data = data_transformation(df)
            train_model(scaled_selected_data, k)

    st.write("\n")
    st.write("\n")
    st.header("Final Model and Cluster Description", help="This module calculates the final model given value of K, assigns clusters to each client, and creates \
        descriptions of each clusters based on modeling and reporting variables")
    # Ask for the final value of K
    st.markdown('<h3 style="font-size: 20px;">Enter the final value of K</h3>', unsafe_allow_html=True)
    final_k = st.number_input(" ", min_value=2, max_value=30, step=1)

    if st.button("Calculate Final Model"):
        with st.spinner("Calculating..."):
            selected_model_columns = global_dictionary['selected_model_columns']
            selected_reporting_columns = global_dictionary['selected_reporting_columns']
            df = global_dictionary['dataframe']
            selected_df = df[['PERS_ID'] + selected_model_columns + selected_reporting_columns]
            selected_df = selected_df.fillna(0)
            # Run the final K-means model
            final_df = run_kmeans(selected_df[selected_model_columns], final_k)
            final_df = pd.concat([selected_df['PERS_ID'], final_df, selected_df[selected_reporting_columns]], axis=1)
            # Calculate average values of variables for each cluster
            avg_df = final_df[selected_model_columns + ['Cluster']].groupby('Cluster').describe()
            avg_df = round(avg_df.transpose(), 2)
            # Convert MultiIndex to Index
            avg_df.reset_index(inplace=True)
            # define first row
            columns_to_slice = [e for e in range(final_k)]
            first_row_df = avg_df[avg_df['level_1'].str.contains('count')][columns_to_slice].reset_index(drop=True)
            # print(first_row_df)
            first_row = first_row_df.iloc[0]
            # Delete count and std measures for each variable
            avg_df = avg_df[~avg_df['level_1'].str.contains('count')]
            avg_df = avg_df[~avg_df['level_1'].str.contains('std')]
            # Reset the index to convert it into columns
            avg_df.reset_index(inplace=True)

            # Combine Level:0 and Level:1 into a single column
            avg_df['Combined'] = avg_df['level_0'].astype(str) + '_' + avg_df['level_1'].astype(str)

            # Set the Combined column as the new index
            avg_df.set_index('Combined', inplace=True)

            # Drop the original Level:0 and Level:1 columns
            avg_df.drop(columns=['level_0', 'level_1', 'index'], inplace=True)

            # Add new countshare column
            new_row_count_share = round(100*first_row / sum(first_row), 2)
            new_row_count_share = pd.DataFrame([new_row_count_share], columns=avg_df.columns)
            avg_df = pd.concat([new_row_count_share, avg_df])
            avg_df.rename({0: 'CountShare %'}, axis='index', inplace = True)

            # Add new count column
            new_row_count = pd.DataFrame([first_row], columns=avg_df.columns)
            avg_df = pd.concat([new_row_count, avg_df])
            avg_df.rename({0: 'Count'}, axis='index', inplace = True)
            #########################
            st.write("\n")
            # Show the final dataframe with average values
            st.write("Final Clustered Dataframe (Average Values):")
            st.dataframe(avg_df, use_container_width = True)

            # Final result downloaders
            avg_df.to_csv('csv_descr.csv')
            final_df.to_csv('csv_client.csv')

            # Zip files
            csv_files = ['csv_descr.csv', 'csv_client.csv']
            with zipfile.ZipFile('output.zip', 'w') as zipf:
                for csv_file in csv_files:
                    # Add each EXCEL file to the ZIP
                    zipf.write(csv_file)

            # Button for downloading final zip file
            with open("output.zip", "rb") as zip_file:
                st.download_button(
                label="Download Final Results",
                data=zip_file,
                file_name='FinalResults.zip',
                mime='application/zip',
                )

if __name__ == '__main__':
    main()
