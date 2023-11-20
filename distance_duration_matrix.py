import json
import requests
import pandas as pd
import numpy as np

# Read employee details from Excel file
df1 = pd.read_excel('Sample-Data.xlsx', sheet_name='Emp Details')

# Specify office location as a string of latitude and longitude
office_loc = '28.4980845,77.40357'

# Define API-related variables
host = 'https://gisnew.smart24x7.com/'
api = 'table/v1/driving/'
api_url = host + api

# Create unique NodeIDs for each employee
df1['NodeID'] = 'EMP' + df1['Employee_System_ID'].astype(str)

# Extract relevant columns and handle office location
df = df1.set_index('NodeID')[['Emp_Pick_LatLong']].copy()
df.loc['OFC000', 'Emp_Pick_LatLong'] = office_loc

# Split latitude and longitude, convert to float
df[['lat', 'long']] = df['Emp_Pick_LatLong'].str.split(',', n=1, expand=True)
df['lat'] = df['lat'].astype(np.float64)
df['long'] = df['long'].astype(np.float64)

# Set the work location and reorder the DataFrame
work_id = 'OFC000'
new_idx = [work_id] + df.index.difference([work_id]).to_list()
df = df.loc[new_idx, :]

# Initialize empty distance and duration matrices
distance_matrix = pd.DataFrame()
duration_matrix = pd.DataFrame()

# Iterate through the data in chunks of 100 to overcome API limitations
for i in range(0, len(df), 100):
    temp_distance_matrix = pd.DataFrame()
    temp_duration_matrix = pd.DataFrame()
    for j in range(0, len(df), 100):
        m = i + 100
        n = j + 100

        # Prepare source and destination points for API request
        source_pts = ";".join(df[i:m].apply(lambda row: f"{round(row['long'], 6)},{round(row['lat'], 6)}", axis=1))
        destination_pts = ";".join(df[j:n].apply(lambda row: f"{round(row['long'], 6)},{round(row['lat'], 6)}", axis=1))
        coord_pts = source_pts + ';' + destination_pts

        # Prepare sources and destinations indices for API request
        destinations = ';'.join([str(i) for i in range(len(df[i:m]), len(df[i:m]) + len(df[j:n]))])
        sources = ';'.join([str(i) for i in range(len(df[i:m]))])

        # Create API request URL
        url = f'{api_url}{coord_pts}?sources={sources}&destinations={destinations}&annotations=distance,duration'
        
        # Make API request
        resp = requests.get(url)
        output = json.loads(resp.text)

        # Extract and concatenate distance and duration matrices for the current chunk
        sources_idx = df[i:m].index.tolist()
        dest_idx = df[j:n].index.tolist()
        temp_output_dist = pd.DataFrame(output['distances'], columns=dest_idx, index=sources_idx)
        temp_output_dur = pd.DataFrame(output['durations'], columns=dest_idx, index=sources_idx)
        temp_distance_matrix = pd.concat([temp_distance_matrix, temp_output_dist], axis=1)
        temp_duration_matrix = pd.concat([temp_duration_matrix, temp_output_dur], axis=1)

    # Concatenate the matrices obtained for the current chunk
    distance_matrix = pd.concat([distance_matrix, temp_distance_matrix])
    duration_matrix = pd.concat([duration_matrix, temp_duration_matrix])

# Save the obtained matrices to CSV files
distance_matrix.to_csv('SampleDataDistanceMatrix.csv')
duration_matrix.to_csv('SampleDatDurationMatrix.csv')
