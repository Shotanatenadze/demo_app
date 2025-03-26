# Project Name

Brief description of your project.

## Table of Contents

- [Installation](#installation)
- [Project Structure](#project-structure)
- [Files Description](#files-description)
- [Functions Description](#functions-description)
- [Usage](#usage)

## Installation

**Prerequisites:**
- Python 3.9
- Pip (Python Package Installer)

To set up the project, follow these steps:

- activating virtual environment: 
\```bash
d:/AutomaticClustering/auto_cluster_env/Scripts/Activate.ps1
\```

- install required libraries
\```bash
pip install -r requirements.txt
\```

## Project Structure

```
.
-- app_new.py
-- utilities.py
-- variables.json
-- requirements.txt
-- ClusteringApp.bat
-- queries/
   -- query1.sql
   -- query2.sql
-- data/
    -- file_1.feather
    -- file_2.feather
```

## Files Description

- `app_new.py`: Main script to run the streamlit application locally with Google Chrome
- `utilities.py`: Script that containt utility functions for the app
- `queries/`: Folder with SQL queries needed for variable calculation
- `data/`: Folder with previous month's datasets

## Functions Description


### `utilities.py`

- `def run_kmeans(df, k):`: given dataframe and value of k, the function estimates K-Means models
- `def connect_oracle_(query):`: Oracle DB connector
- `def execute_script_part(connection, script_part):`: Given Oracle DB connection and sql query it runs the scripts against DB
- `def create_views():`: Creating views for variable calculation
- `def data_downloader(query):`: Downloading final data from Oracle DB given the query

## Usage

To use the main functionality of the project, run:

\```bash
streamlit run app_new.py
\```

or you can just click ClusteringApp.bat file
