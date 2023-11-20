# Employee Transportation Optimization

This repository contains three Python scripts designed to optimize employee transportation using the Google OR-Tools library and Google Maps Distance Matrix API. Below is an overview of each script and instructions on how to use them.


## 1. `distance_duration_matrix.py`

### Description
The `distance_duration_matrix.py` script fetches distance and duration matrices for employee locations using the Google Maps Distance Matrix API. It reads employee details from an Excel file, sends API requests in chunks, and saves the obtained matrices as CSV files.

### Usage
1. Import the necessary libraries: pandas, NumPy, and requests.
2. Read employee details from an Excel file, specify the office location, and define API-related variables.
3. Create unique NodeIDs for each employee, handle office location, and split latitude and longitude.
4. Initialize empty distance and duration matrices.
5. Iterate through the data in chunks, prepare API requests, and fetch distance and duration matrices.
6. Save the obtained matrices as CSV files ('SampleDataDistanceMatrix.csv' and 'SampleDataDurationMatrix.csv').


## 2. `main.py`

### Description
The `main.py` script implements a routing optimization solution using the Google OR-Tools library. It aims to efficiently assign employees to vehicles for transportation, considering various constraints such as distance, time, and vehicle capacity.

### Usage
1. Import the necessary libraries: OR-Tools, NumPy, and pandas.
2. Initialize the `Routing` class with employee and route details.
3. Configure the routing parameters such as distance, duration, cost, and demand callbacks.
4. Generate initial routes, create a routing model, and set up dimensions for distance, duration, and vehicle capacity.
5. Solve the routing problem and print the optimized solution.
