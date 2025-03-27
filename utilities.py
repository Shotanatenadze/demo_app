import pandas as pd
import streamlit as st
# import cx_Oracle 
from sklearn.cluster import KMeans
import time
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
from pandas.api.types import (
    is_categorical_dtype,
    is_datetime64_any_dtype,
    is_numeric_dtype,
    is_object_dtype,
)

# Function for filtering dataframe (At this moment this function isn't being used)
def filter_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds a UI on top of a dataframe to let viewers filter columns

    Args:
        df (pd.DataFrame): Original dataframe

    Returns:
        pd.DataFrame: Filtered dataframe
    """
    modify = st.checkbox("Add filters")

    if not modify:
        return df

    df = df.copy()

    columns = ['MONTHLY_INCOME',
    'AGE',
    'GENDER',
    'PREF_CITY',
    'RES_FLAG',
    'PAYROLL_TP',
    'SOC_SEG',
    'TENURE',
    'IS_INT_BNK_USER',
    'IS_MOB_BNK_USER',
    'AVG_MB_IB_TRN_CNT',
    'AVG_POS_TRN_CNT',
    'AVG_ATM_TRN_CNT']
    df['REGION'].fillna("თბილისი", inplace=True)
    df['PREF_CITY'].fillna("თბილისი", inplace=True)
    # Try to convert datetimes into a standard format (datetime, no timezone)
    for col in columns:
        if is_object_dtype(df[col]):
            try:
                df[col] = pd.to_datetime(df[col])
            except Exception:
                pass

        if is_datetime64_any_dtype(df[col]):
            df[col] = df[col].dt.tz_localize(None)

    modification_container = st.container()

    with modification_container:
        to_filter_columns = st.multiselect("Filter dataframe on", columns)
        for column in to_filter_columns:
            left, right = st.columns((1, 20))
            left.write("↳")
            # Treat columns with < 10 unique values as categorical
            if (is_categorical_dtype(df[column]) or df[column].nunique() < 88) and column != "AVG_ATM_TRN_CNT":
                user_cat_input = right.multiselect(
                    f"Values for: {column}",
                    df[column].unique(),
                    # default=list(df[column].unique()),
                )
                df = df[df[column].isin(user_cat_input)]
            elif is_numeric_dtype(df[column]):
                _min = 0.0 #min(0.0, float(df[column].min()))
                _max = float(df[column].max())
                step = (_max - _min) / 100
                user_num_input = right.slider(
                    f"Values for: {column}",
                    _min,
                    _max,
                    (_min, _max),
                    step=step,
                )
                df = df[df[column].between(*user_num_input)]
            elif is_datetime64_any_dtype(df[column]):
                user_date_input = right.date_input(
                    f"Values for: {column}",
                    value=(
                        df[column].min(),
                        df[column].max(),
                    ),
                )
                if len(user_date_input) == 2:
                    user_date_input = tuple(map(pd.to_datetime, user_date_input))
                    start_date, end_date = user_date_input
                    df = df.loc[df[column].between(start_date, end_date)]
            else:
                user_text_input = right.text_input(
                    f"Substring or regex in {column}",
                )
                if user_text_input:
                    df = df[df[column].str.contains(user_text_input)]

    return df


# Function to run K-means algorithm
def run_kmeans(df, k):
    # Standardize the data
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(df)

    # Run K-means algorithm
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(scaled_data)

    # Get cluster labels
    labels = kmeans.labels_

    # Add cluster labels to the dataframe
    df['Cluster'] = labels

    return df


# Oracle database connector function
def connect_oracle():
     
    #Authentication
    user = ""
    pwd = ""
    dsn = ""
    
    #Connection
    connection = cx_Oracle.connect(user=user, password=pwd, dsn=dsn, encoding="UTF-8")
    return connection

# script execution against SQL DB
def execute_script_part(connection, script_part):
    cursor = connection.cursor()
    cursor.execute(script_part)
    cursor.close()

# Function creates views that are needed for model and reporting variable calculation
# by default date filter is based on the previous month at any given moment in time
def create_views():

    connection = connect_oracle()
    
    sql_script = """   
    CREATE OR REPLACE VIEW APP_TRN_TABLE_TEMP_VIEW AS (
    SELECT PERS_ID, SNAP_DATE, TRN_CLASS_DESCR, CNT, AMT
    FROM ADS_ETL_OWNER.ODS_OPTIO_TRN(TO_DATE(LAST_DAY(ADD_MONTHS(SYSDATE, -1))))
    WHERE SUBSTR(P_CODE, 1, 1) = 'P' 
    UNION ALL
    SELECT PERS_ID, SNAP_DATE, TRN_CLASS_DESCR, CNT, AMT
    FROM ADS_ETL_OWNER.ODS_OPTIO_TRN(TO_DATE(LAST_DAY(ADD_MONTHS(SYSDATE, -2))))
    WHERE SUBSTR(P_CODE, 1, 1) = 'P' 
    UNION ALL
    SELECT PERS_ID, SNAP_DATE, TRN_CLASS_DESCR, CNT, AMT
    FROM ADS_ETL_OWNER.ODS_OPTIO_TRN(TO_DATE(LAST_DAY(ADD_MONTHS(SYSDATE, -3))))
    WHERE SUBSTR(P_CODE, 1, 1) = 'P');

    CREATE OR REPLACE VIEW APP_SPEND_CATEG_SUM_RAW_VIEW AS (
    SELECT * FROM (
    SELECT PERS_ID AS PERS_ID_01, TRN_CLASS_DESCR, CNT, AMT
    FROM APP_TRN_TABLE_TEMP_VIEW
    )
    PIVOT (
        SUM(CNT) AS SUM_CNT, SUM(AMT) AS SUM_AMT
        FOR TRN_CLASS_DESCR IN 
        ('Healthcare & Beauty' AS Healthcare_Beauty, 
        'Food & Groceries' AS Food_Groceries, 
        'Shopping' AS Shopping, 
        'Leisure & Entertainment' AS Leisure_Entertainment, 
        'Uncategorized' AS Uncategorized, 
        'Bars & Restaurants' AS Bars_Restaurants, 
        'Auto & Transport' AS Auto_Transport, 
        'Home & Rent' AS Home_Rent, 
        'Education' AS Education,
        'Travel & Holidays' AS Travel_Holidays, 
        'Family & Friends' AS Family_Friends, 
        'Other Spending' AS Other_Spending, 
        'Business Services' AS Business_Services)
    ) PVT
    );

    CREATE OR REPLACE VIEW APP_SPEND_CATEG_AVG_VIEW AS (
    SELECT
    PERS_ID_01,
    HEALTHCARE_BEAUTY_SUM_CNT/3 AS HEALTHCARE_BEAUTY_AVG_CNT,
    HEALTHCARE_BEAUTY_SUM_AMT/3 AS HEALTHCARE_BEAUTY_AVG_AMT,
    FOOD_GROCERIES_SUM_CNT/3 AS FOOD_GROCERIES_AVG_CNT,
    FOOD_GROCERIES_SUM_AMT/3 AS FOOD_GROCERIES_AVG_AMT,
    SHOPPING_SUM_CNT/3 AS SHOPPING_AVG_CNT,
    SHOPPING_SUM_AMT/3 AS SHOPPING_AVG_AMT,
    LEISURE_ENTERTAINMENT_SUM_CNT/3 AS LEIS_ENTERT_AVG_CNT,
    LEISURE_ENTERTAINMENT_SUM_AMT/3 AS LEIS_ENTERT_AVG_AMT,
    UNCATEGORIZED_SUM_CNT/3 AS UNCATEGORIZED_AVG_CNT,
    UNCATEGORIZED_SUM_AMT/3 AS UNCATEGORIZED_AVG_AMT,
    BARS_RESTAURANTS_SUM_CNT/3 AS BARS_RESTAURANTS_AVG_CNT,
    BARS_RESTAURANTS_SUM_AMT/3 AS BARS_RESTAURANTS_AVG_AMT,
    AUTO_TRANSPORT_SUM_CNT/3 AS AUTO_TRANSPORT_AVG_CNT,
    AUTO_TRANSPORT_SUM_AMT/3 AS AUTO_TRANSPORT_AVG_AMT,
    HOME_RENT_SUM_CNT/3 AS HOME_RENT_AVG_CNT,
    HOME_RENT_SUM_AMT/3 AS HOME_RENT_AVG_AMT,
    EDUCATION_SUM_CNT/3 AS EDUCATION_AVG_CNT,
    EDUCATION_SUM_AMT/3 AS EDUCATION_AVG_AMT,
    TRAVEL_HOLIDAYS_SUM_CNT/3 AS TRAVEL_HOLIDAYS_AVG_CNT,
    TRAVEL_HOLIDAYS_SUM_AMT/3 AS TRAVEL_HOLIDAYS_AVG_AMT,
    FAMILY_FRIENDS_SUM_CNT/3 AS FAMILY_FRIENDS_AVG_CNT,
    FAMILY_FRIENDS_SUM_AMT/3 AS FAMILY_FRIENDS_AVG_AMT,
    OTHER_SPENDING_SUM_CNT/3 AS OTHER_SPENDING_AVG_CNT,
    OTHER_SPENDING_SUM_AMT/3 AS OTHER_SPENDING_AVG_AMT,
    BUSINESS_SERVICES_SUM_CNT/3 AS BUSINESS_SERVICES_AVG_CNT,
    BUSINESS_SERVICES_SUM_AMT/3 AS BUSINESS_SERVICES_AVG_AMT
    FROM APP_SPEND_CATEG_SUM_RAW_VIEW
    );

    CREATE OR REPLACE VIEW APP_INC_SUB_CATG_SUM_RAW_VIEW AS (
    SELECT * FROM (
    SELECT PERS_ID AS PERS_ID_02, TRN_CLASS_DESCR, CNT, AMT
    FROM APP_TRN_TABLE_TEMP_VIEW
    )
    PIVOT (
        SUM(CNT) AS SUM_CNT, SUM(AMT) AS SUM_AMT
        FOR TRN_CLASS_DESCR IN 
        (
        'FX - Sell' AS FX_SELL,
        'Social Care' AS SOCIAL_CARE,
        'Bonus' AS BONUS,
        'Income from Deposits' AS INCOME_FROM_DEPOSITS,
        'Retirement' AS RETIREMENT,
        'Branch' AS BRANCH,
        'Salary' AS SALARY,
        'Incoming Internal Transfer Between Own' AS INC_INT_TRANSF_OWN,
        'Dividend' AS DIVIDEND,
        'FX - Buy' AS FX_BUY,
        'Other Income' AS OTHER_INCOME,
        'Cashback' AS CASHBACK,
        'ATM' AS ATM,
        'Outgoing Internal Transfer Between Own' AS OUT_INT_TRANSF_OWN
        )
    ) PVT
    );

    CREATE OR REPLACE VIEW APP_INC_CATEG_AVG_VIEW AS (
    SELECT 
    PERS_ID_02,
    FX_SELL_SUM_CNT / 3 AS FX_SELL_SUB_AVG_CNT,
    FX_SELL_SUM_AMT / 3 AS FX_SELL_SUB_AVG_AMT,
    SOCIAL_CARE_SUM_CNT / 3 AS SOCIAL_CARE_AVG_CNT,
    SOCIAL_CARE_SUM_AMT / 3 AS SOCIAL_CARE_AVG_AMT,
    BONUS_SUM_CNT / 3 AS BONUS_AVG_CNT,
    BONUS_SUM_AMT / 3 AS BONUS_AVG_AMT,
    INCOME_FROM_DEPOSITS_SUM_CNT / 3 AS INCOME_FROM_DEPOSITS_AVG_CNT,
    INCOME_FROM_DEPOSITS_SUM_AMT / 3 AS INCOME_FROM_DEPOSITS_AVG_AMT,
    RETIREMENT_SUM_CNT / 3 AS RETIREMENT_AVG_CNT,
    RETIREMENT_SUM_AMT / 3 AS RETIREMENT_AVG_AMT,
    BRANCH_SUM_CNT / 3 AS BRANCH_AVG_CNT,
    BRANCH_SUM_AMT / 3 AS BRANCH_AVG_AMT,
    SALARY_SUM_CNT / 3 AS SALARY_AVG_CNT,
    SALARY_SUM_AMT / 3 AS SALARY_AVG_AMT,
    INC_INT_TRANSF_OWN_SUM_CNT / 3 AS INC_INT_TRANSF_OWN_AVG_CNT,
    INC_INT_TRANSF_OWN_SUM_AMT / 3 AS INC_INT_TRANSF_OWN_AVG_AMT,
    DIVIDEND_SUM_CNT / 3 AS DIVIDEND_AVG_CNT,
    DIVIDEND_SUM_AMT / 3 AS DIVIDEND_AVG_AMT,
    FX_BUY_SUM_CNT / 3 AS FX_BUY_SUB_AVG_CNT,
    FX_BUY_SUM_AMT / 3 AS FX_BUY_SUB_AVG_AMT,
    OTHER_INCOME_SUM_CNT / 3 AS OTHER_INCOME_AVG_CNT,
    OTHER_INCOME_SUM_AMT / 3 AS OTHER_INCOME_AVG_AMT,
    CASHBACK_SUM_CNT / 3 AS CASHBACK_AVG_CNT,
    CASHBACK_SUM_AMT / 3 AS CASHBACK_AVG_AMT,
    ATM_SUM_CNT / 3 AS ATM_AVG_CNT,
    ATM_SUM_AMT / 3 AS ATM_AVG_AMT,
    OUT_INT_TRANSF_OWN_SUM_CNT / 3 AS OUT_INT_TRANSF_OWN_AVG_CNT,
    OUT_INT_TRANSF_OWN_SUM_AMT / 3 AS OUT_INT_TRANSF_OWN_AVG_AMT
    FROM APP_INC_SUB_CATG_SUM_RAW_VIEW
    );


    CREATE OR REPLACE VIEW APP_INIT_TABLE_VIEW AS (
    SELECT PERS_ID, UNI_PT_KEY, PREF_CITY, SEG_TP, NEW_SEG_TP, SOC_SEG, DUAL_BANKER, CRINFO_CHECKS, RES_FLAG, FAV_BRNCH_VISIT_CNT, FAV_ATM_VISIT_COUNT
    FROM ADS_ETL_OWNER.FINREP_OPTIO_CUSTOMERS
        WHERE SUBSTR(P_CODE, 1, 1) = 'P' 
                AND DEL_FLAG = 0 
                AND PT_STAT_DESCR = 'Active'
                AND UNI_PT_KEY IN (SELECT UNI_PT_KEY FROM ADS_ETL_OWNER.FINREP_OPTIO_CUSTOMERS_FACT 
                                            WHERE SNAP_DATE = TO_DATE(LAST_DAY(ADD_MONTHS(SYSDATE, -1))) AND M_ACT_FLAG = 1)
                                            );


    CREATE OR REPLACE VIEW APP_OPTIO_SEGM_VIEW AS (
    SELECT PERS_ID AS PERS_ID_03, AGE, GENDER, TENURE, REGION, PAYROLL_TP
    FROM ADS_ETL_OWNER.FINREP_OPTIO_SEGM);


    CREATE OR REPLACE VIEW APP_FACT_VARIABLES_VIEW AS (
    SELECT UNI_PT_KEY AS UNI_PT_KEY_01, SNAP_DATE, T_ACCOUNTS_BAL, T_SAVINGS_BAL, MONTHLY_INCOME, 
        SIXMONTHS_TURNOVER, LOAN_AMOUNT, DEPO_AMOUNT
    FROM ADS_ETL_OWNER.FINREP_OPTIO_CUSTOMERS_FACT
    WHERE SNAP_DATE = TO_DATE(LAST_DAY(ADD_MONTHS(SYSDATE, -1))));


    CREATE OR REPLACE VIEW APP_SEGM_FACT_VARIABLES_VIEW AS (
    SELECT PERS_ID AS PERS_ID_04, ACC_AVG_MIN_BAL, ACC_AVG_MAX_BAL, DEPO_AVG_MIN_BAL, DEPO_AVG_MAX_BAL,
        CARD_TRN_AMT_SUM_MIN_1, ACC_TRN_AMT_SUM_MIN_1, ACC_REST_START_MIN_1, ACC_REST_END_MIN_1, 
        CARD_ATM_TRN_LAST_3, CARD_ATM_TRN_LAST_CNT_3, DB_CARD_FIRST_BAL_MIN_1, DB_CARD_LAST_BAL_MIN_1,
        LAST_LOAN_AMT, LAST_LOAN_DPD, ACT_TOT_LOAN_CNT, ACT_TOT_LOAN_AMT, ACT_TOT_DPD, ACT_TOT_CNT_DPD,
        CLO_TOT_LOAN_CNT, CLO_TOT_LOAN_AMT, CLO_TOT_DPD, CLO_TOT_CNT_DPD,
        IS_INT_BNK_USER, IS_MOB_BNK_USER, AVG_LOGIN_CNT_MOB_BNK, AVG_LOGIN_CNT_INT_BNK,
        AVG_TRN_CNT_INT_BNK, AVG_TRN_CNT_MOB_BNK
    FROM ADS_ETL_OWNER.FINREP_OPTIO_SEGM_FACT
    WHERE SNAP_DATE = TO_DATE(LAST_DAY(ADD_MONTHS(SYSDATE, -1))));

    CREATE OR REPLACE VIEW APP_TRN_CHANNELS_DATA_VIEW AS 
    ( 
    SELECT  T_0.PERS_ID AS PERS_ID_05, T_1.AVG_MB_IB_TRN_CNT, T_2.AVG_POS_TRN_CNT, T_3.AVG_ATM_TRN_CNT, 
    T_4.AVG_BRNCH_VISIT_CNT, T_1.AVG_MB_IB_TRN_AMT, T_2.AVG_POS_TRN_AMT, T_3.AVG_ATM_TRN_AMT, T_4.AVG_BRNCH_TRN_AMT
    FROM APP_INIT_TABLE_VIEW T_0
    LEFT JOIN (
    --1. TRN: INT AND MOB BANK (145,019  Rows)
    SELECT PT_ID, COUNT(*)/3 AS AVG_MB_IB_TRN_CNT, ROUND(SUM(TRN_LCCY_AMT)/3, 1) AS AVG_MB_IB_TRN_AMT FROM
    (
    SELECT TRN_DATE, PT_ID, PT_CODE, DEL_FLAG, TRN_LCCY_AMT
    FROM ads_L2_eventdm_owneR.EVENTDM_PT_EVENT_FACT
    WHERE (TRN_DATE BETWEEN TO_DATE(LAST_DAY(ADD_MONTHS(SYSDATE, -3))) AND TO_DATE(LAST_DAY(ADD_MONTHS(SYSDATE, -1)))) 
            AND SUBSTR(PT_CODE, 1, 1) = 'P' 
            AND DEL_FLAG = 0 
            AND SRC_SYS_ID='OMC0' 
            AND (TRN_LCCY_AMT > 0)
            AND PROD_TP != 'ქეშბექი'
            AND ENTR_STAT_DESCR IN ('დადასტურებული (დამუშავებული)', 'ახალი', 'დადასტურებული (დაუმუშავებელი)', 'დასრულებული განაცხადები', 'დაუსრულებელი განაცხადები', 'დამტკიცებულია')
    )
    GROUP BY PT_ID ) T_1 ON T_0.PERS_ID = T_1.PT_ID
    LEFT JOIN (
    --2. TRN: POS (236,448  Rows)
    SELECT PT_ID, COUNT(*)/3 AS AVG_POS_TRN_CNT, ROUND(SUM(TRN_LCCY_AMT)/3, 1) AS AVG_POS_TRN_AMT FROM
    (   
    SELECT TRN_DATE, PT_ID, PT_CODE, DEL_FLAG, TRN_LCCY_AMT
    FROM ads_L2_eventdm_owneR.EVENTDM_PT_EVENT_FACT
    WHERE (TRN_DATE BETWEEN TO_DATE(LAST_DAY(ADD_MONTHS(SYSDATE, -3))) AND TO_DATE(LAST_DAY(ADD_MONTHS(SYSDATE, -1)))) 
            AND SUBSTR(PT_CODE, 1, 1) = 'P' 
            AND DEL_FLAG = 0 
            AND SRC_SYS_ID = 'RPR0' 
            AND (CNL_KEY IN (21, 22))  
            AND (TRN_LCCY_AMT > 0)
            AND PROD_TP != 'ქეშბექი'
            AND ENTR_STAT_DESCR IN ('დადასტურებული (დამუშავებული)', 'ახალი', 'დადასტურებული (დაუმუშავებელი)', 'დასრულებული განაცხადები', 'დაუსრულებელი განაცხადები', 'დამტკიცებულია')
    )
    GROUP BY PT_ID ) T_2 ON T_0.PERS_ID = T_2.PT_ID
    LEFT JOIN (
    --3. TRN: ATM (1,016,779  Rows)
    SELECT PT_ID, COUNT(*)/3 AS AVG_ATM_TRN_CNT, ROUND(SUM(TRN_LCCY_AMT)/3, 1) AS AVG_ATM_TRN_AMT FROM
    (
    SELECT TRN_DATE, PT_ID, PT_CODE, DEL_FLAG, TRN_LCCY_AMT
    FROM ads_L2_eventdm_owneR.EVENTDM_PT_EVENT_FACT
    WHERE (TRN_DATE BETWEEN TO_DATE(LAST_DAY(ADD_MONTHS(SYSDATE, -3))) AND TO_DATE(LAST_DAY(ADD_MONTHS(SYSDATE, -1)))) 
            AND SUBSTR(PT_CODE, 1, 1) = 'P' 
            AND DEL_FLAG = 0 
            AND SRC_SYS_ID = 'RPR0' 
            AND CNL_KEY = 17 
            AND entr_sub_tp_descr = 'თანხის გატანა / ლიბერთი / ბანკომატი' 
            AND (TRN_LCCY_AMT > 0)
            AND PROD_TP != 'ქეშბექი'
            AND ENTR_STAT_DESCR IN ('დადასტურებული (დამუშავებული)', 'ახალი', 'დადასტურებული (დაუმუშავებელი)', 'დასრულებული განაცხადები', 'დაუსრულებელი განაცხადები', 'დამტკიცებულია')
    )
    GROUP BY PT_ID) T_3 ON T_0.PERS_ID = T_3.PT_ID
    LEFT JOIN (
    --4. TRN: COUNT OF VISITS TO BRANCHES (522,499  Rows)
    SELECT PT_ID, COUNT(*)/3 AS AVG_BRNCH_VISIT_CNT, ROUND(SUM(TRN_LCCY_AMT)/3, 1) AS AVG_BRNCH_TRN_AMT FROM 
    (
    SELECT TRN_DATE, PT_ID, PT_CODE, DEL_FLAG, TRN_LCCY_AMT
    from ads_L2_eventdm_owneR.EVENTDM_EMP_EVENT_FACT
    where UNI_PT_KEY NOT IN (-1, -2)
                        AND POS_ID != '650'
                        AND SRC_SYS_ID NOT IN ('BLB0', 'RSL0')
                        AND POS_GEO_REG_DESCR != 'ვირტუალური'
                        AND POS_KEY != 596
                        AND SUBSTR(PT_CODE, 1, 1) = 'P'
                        AND DEL_FLAG = 0
                        AND (TRN_DATE BETWEEN TO_DATE(LAST_DAY(ADD_MONTHS(SYSDATE, -3))) AND TO_DATE(LAST_DAY(ADD_MONTHS(SYSDATE, -1))))
                        AND (TRN_LCCY_AMT > 0)
    AND (EVENT_SUB_TP_DESCR IN (
    'ლარის გადარიცხვა ბანკის შიგნით',
    'ლიბერთი ექსპრესი (გაგზავნა)',
    'დასუფთავების გადახდა',
    'შპს კავკასიის ავტოიმპორტი',
    'ანგარიშიდან თანხის გატანა (ფიზიკური პირი)',
    'დამატებითი ბარათით თანხის გატანა',
    'ექსპრეს მანი (გაგზავნა)',
    'კლიენტის ანგარიშიდან თანხის გადარიცხვა GEL',
    'კონტაქტი (გაგზავნა)',
    'სხვა სახის გადასახადები',
    'ავტობუსის ჯარიმის გადახდა',
    'ჯარიმის საგზაო მოძრაობის წესების დარღვევისთვის გადახდა',
    'კლიენტის ანგარიშიდან სავალუტო  გადარიცხვა ',
    'თანხის გაცემა სოციალური ანგარიშიდან',
    'კრედიტის წინსწრებით დაფარვა',
    'მანიგრამი (გაგზავნა)',
    'რია (გაგზავნა)',
    'სესხის გაცემა',
    'ანგარიშიდან თანხის გატანა',
    'კომუნალური დავალიანების გადახდა',
    'მომსახურების სააგენტოს გადახდები',
    'ინტელექსპრესი (საბერძნეთი) (გაგზავნა)',
    'საგადასახადო ინსპექცია',
    'ხე-ტყის დამუშავების გადასახადის გადახდა',
    'უცხოური ვალუტის გადარიცხვა ბანკის შიგნით',
    'ვესტერნ იუნიონი (გაგზავნა)',
    '125 მუხლი',
    'ფიზიკური პირის ანგარიშებიდან ბანკთაშორისი  სავალუტო გადარიცხვები',
    'სესხის დაფარვა',
    'უცხოური ავტოტრანსპორტის დაზღვევა',
    'ლარის გადარიცხვა ბანკის გარეთ',
    'ზოლოტაია კორონა (გაგზავნა)',
    'ველსენდი (გაგზავნა)',
    'არასტანდარტული პროვაიდერის გადახდა',
    'საბიუჯეტო გადარიცხვა',
    'ვალუტის გადარიცხვა ბანკის გარეთ',
    'პროვაიდერის ანგარიშიდან თანხის გატანა',
    'უნისტრიმი (გაგზავნა)'
    )
    )                                     
    )                    
    GROUP BY PT_ID) T_4 ON T_0.PERS_ID = T_4.PT_ID
    );

    CREATE OR REPLACE VIEW APP_FINAL_LARGE_VIEW AS (
    SELECT T_1.*, T_2.*, T_3.*, T_4.*, T_5.*, T_6.*, T_7.*
    FROM APP_INIT_TABLE_VIEW T_1
    LEFT JOIN (
        SELECT *
        FROM APP_SPEND_CATEG_AVG_VIEW) T_2 ON T_1.PERS_ID = T_2.PERS_ID_01
    LEFT JOIN (
        SELECT *
        FROM APP_INC_CATEG_AVG_VIEW) T_3 ON T_1.PERS_ID = T_3.PERS_ID_02
    LEFT JOIN (
        SELECT *
        FROM APP_OPTIO_SEGM_VIEW) T_4 ON T_1.PERS_ID = T_4.PERS_ID_03
    LEFT JOIN (
        SELECT *
        FROM APP_FACT_VARIABLES_VIEW) T_5 ON T_1.UNI_PT_KEY = T_5.UNI_PT_KEY_01
    LEFT JOIN (
        SELECT *
        FROM APP_SEGM_FACT_VARIABLES_VIEW) T_6 ON T_1.PERS_ID = T_6.PERS_ID_04
    LEFT JOIN (
        SELECT *
        FROM APP_TRN_CHANNELS_DATA_VIEW) T_7 ON T_1.PERS_ID = T_7.PERS_ID_05
    )
    """
    
    script_parts = sql_script.split(';') 
    
    for script_part in script_parts:
        execute_script_part(connection, script_part)
        print("VIEW CREATED")
    
    connection.commit()
    connection.close()


def connect_oracle_(query):
     
    #Authentication
    user = ""
    pwd = ""
    dsn = ""
    
    #Connection
    connection = cx_Oracle.connect(user=user, password=pwd, dsn=dsn, encoding="UTF-8")
    cur = connection.cursor()
    cur.execute(query)
    return cur

# Query execution function
def data_downloader(query):
    #Fetch data from Oracle
    fetched_data = connect_oracle_(query)
    
    #Initialize column names
    column_names = [row[0] for row in fetched_data.description]
    
    #Initialize empty list to store dictionaries of fetched data
    initial_list_1 = []
    nrow = 0
    start_time = time.time()
    for result in tqdm(fetched_data):
        dict_data = dict(zip(column_names, result))
        initial_list_1.append(dict_data)
        nrow += 1
        if nrow%100000 == 0:
            end_time = time.time()
            print("There're {} rows in a dataframe".format(nrow))
            print("{} seconds were elapsed".format(end_time - start_time))
            
    print("Finishing a job with Oracle. There are {} rows".format(nrow))
    
    #Transform list of dictionaries to Pandas dataframe object
    start = time.time()
    df = pd.DataFrame.from_dict(initial_list_1)
    end = time.time()
    print("{} seconds were needed to create a final DataFrame".format(end-start))
    
    return df

# Final data downloader function
def download_data():
    # Define query
    query = """
    SELECT *
    FROM APP_FINAL_LARGE_VIEW
    """
    print("STARTING DATA DOWNLOADING")
    # Download data
    df_0 = data_downloader(query)
    print("DATA DOWNLOADED")
    # Select queries
    selected_cols = [
    'PERS_ID',
    'UNI_PT_KEY',
    'AGE',
    'GENDER',
    'TENURE',
    'REGION',
    'PREF_CITY',
    'PAYROLL_TP',
    'SEG_TP',
    'NEW_SEG_TP',
    'SOC_SEG',
    'DUAL_BANKER',
    'CRINFO_CHECKS',
    'RES_FLAG',
    'FAV_BRNCH_VISIT_CNT',
    'FAV_ATM_VISIT_COUNT',
    'HEALTHCARE_BEAUTY_AVG_CNT',
    'HEALTHCARE_BEAUTY_AVG_AMT',
    'FOOD_GROCERIES_AVG_CNT',
    'FOOD_GROCERIES_AVG_AMT',
    'SHOPPING_AVG_CNT',
    'SHOPPING_AVG_AMT',
    'LEIS_ENTERT_AVG_CNT',
    'LEIS_ENTERT_AVG_AMT',
    'UNCATEGORIZED_AVG_CNT',
    'UNCATEGORIZED_AVG_AMT',
    'BARS_RESTAURANTS_AVG_CNT',
    'BARS_RESTAURANTS_AVG_AMT',
    'AUTO_TRANSPORT_AVG_CNT',
    'AUTO_TRANSPORT_AVG_AMT',
    'HOME_RENT_AVG_CNT',
    'HOME_RENT_AVG_AMT',
    'EDUCATION_AVG_CNT',
    'EDUCATION_AVG_AMT',
    'TRAVEL_HOLIDAYS_AVG_CNT',
    'TRAVEL_HOLIDAYS_AVG_AMT',
    'FAMILY_FRIENDS_AVG_CNT',
    'FAMILY_FRIENDS_AVG_AMT',
    'OTHER_SPENDING_AVG_CNT',
    'OTHER_SPENDING_AVG_AMT',
    'BUSINESS_SERVICES_AVG_CNT',
    'BUSINESS_SERVICES_AVG_AMT',
    'FX_SELL_SUB_AVG_CNT',
    'FX_SELL_SUB_AVG_AMT',
    'SOCIAL_CARE_AVG_CNT',
    'SOCIAL_CARE_AVG_AMT',
    'BONUS_AVG_CNT',
    'BONUS_AVG_AMT',
    'INCOME_FROM_DEPOSITS_AVG_CNT',
    'INCOME_FROM_DEPOSITS_AVG_AMT',
    'RETIREMENT_AVG_CNT',
    'RETIREMENT_AVG_AMT',
    'BRANCH_AVG_CNT',
    'BRANCH_AVG_AMT',
    'SALARY_AVG_CNT',
    'SALARY_AVG_AMT',
    'INC_INT_TRANSF_OWN_AVG_CNT',
    'INC_INT_TRANSF_OWN_AVG_AMT',
    'DIVIDEND_AVG_CNT',
    'DIVIDEND_AVG_AMT',
    'FX_BUY_SUB_AVG_CNT',
    'FX_BUY_SUB_AVG_AMT',
    'OTHER_INCOME_AVG_CNT',
    'OTHER_INCOME_AVG_AMT',
    'CASHBACK_AVG_CNT',
    'CASHBACK_AVG_AMT',
    'ATM_AVG_CNT',
    'ATM_AVG_AMT',
    'OUT_INT_TRANSF_OWN_AVG_CNT',
    'OUT_INT_TRANSF_OWN_AVG_AMT',
    'T_ACCOUNTS_BAL',
    'T_SAVINGS_BAL',
    'MONTHLY_INCOME',
    'SIXMONTHS_TURNOVER',
    'LOAN_AMOUNT',
    'DEPO_AMOUNT',
    'ACC_AVG_MIN_BAL',
    'ACC_AVG_MAX_BAL',
    'DEPO_AVG_MIN_BAL',
    'DEPO_AVG_MAX_BAL',
    'CARD_TRN_AMT_SUM_MIN_1',
    'ACC_TRN_AMT_SUM_MIN_1',
    'ACC_REST_START_MIN_1',
    'ACC_REST_END_MIN_1',
    'CARD_ATM_TRN_LAST_3',
    'CARD_ATM_TRN_LAST_CNT_3',
    'DB_CARD_FIRST_BAL_MIN_1',
    'DB_CARD_LAST_BAL_MIN_1',
    'LAST_LOAN_AMT',
    'LAST_LOAN_DPD',
    'ACT_TOT_LOAN_CNT',
    'ACT_TOT_LOAN_AMT',
    'ACT_TOT_DPD',
    'ACT_TOT_CNT_DPD',
    'CLO_TOT_LOAN_CNT',
    'CLO_TOT_LOAN_AMT',
    'CLO_TOT_DPD',
    'CLO_TOT_CNT_DPD',
    'IS_INT_BNK_USER',
    'IS_MOB_BNK_USER',
    'AVG_LOGIN_CNT_MOB_BNK',
    'AVG_LOGIN_CNT_INT_BNK',
    'AVG_TRN_CNT_INT_BNK',
    'AVG_TRN_CNT_MOB_BNK',
    'AVG_MB_IB_TRN_CNT',
    'AVG_POS_TRN_CNT',
    'AVG_ATM_TRN_CNT',
    'AVG_BRNCH_VISIT_CNT',
    'AVG_MB_IB_TRN_AMT',
    'AVG_POS_TRN_AMT',
    'AVG_ATM_TRN_AMT',
    'AVG_BRNCH_TRN_AMT']
    df_1 = df_0[selected_cols]
    return df_1