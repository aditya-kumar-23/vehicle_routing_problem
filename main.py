import pandas as pd
import numpy as np
import route_generator

# Disable pandas chained assignment warning
pd.options.mode.chained_assignment = None  # default='warn'

# Define the route type
route_choice = 'pickup'

# Read the distance matrix, time matrix, employee details, and shift details from Excel files
dist_mat = pd.read_csv('SampleDataDistanceMatrix.csv', index_col=0)
time_mat = pd.read_csv('SampleDataDurationMatrix.csv', index_col=0)
emp_details = pd.read_excel('Sample-Data.xlsx', sheet_name='Emp Details')
emp_shift = pd.read_excel('Sample-Data.xlsx', sheet_name='Schedule')

# Transpose the matrices if the route choice is pickup
if route_choice == 'pickup':
    dist_mat = dist_mat.T
    time_mat = time_mat.T

# Define the vehicle capacity, maximum distance, and maximum time constraints
MaxVehicleCapacity = 4
MaxDist = 48
MaxTime = 120

# Extract unique shift names
shifts = list(emp_shift['Shift Name'].unique())

# Create an empty output DataFrame
output = pd.DataFrame()

# Add employee IDs and set the index to employee IDs
emp_details['NodeID'] = 'EMP' + emp_details['Employee_System_ID'].astype(str)
emp_details = emp_details.set_index('NodeID')
emp_ids = emp_details.index

# Extract employee location and gender information
loc_df = emp_details[['Emp_Pick_LatLong']].copy()
loc_df.loc['OFC000', 'Emp_Pick_LatLong'] = '28.498620,77.404308'
loc_df[['lat', 'long']] = loc_df['Emp_Pick_LatLong'].str.split(',', expand=True).astype(float)
genders = emp_details['Emp_Gender']

# Define the work location ID
work_id = 'OFC000'

# Define the cost per kilometer and cost per minute
cost_per_km = 15
cost_per_min = 0

# Iterate through each shift
for shift in shifts:
    # Extract the employee IDs for the current shift
    this_shift_ids = ('EMP' + emp_shift.loc[emp_shift['Shift Name'] == f'{shift}', 'Employee_System_ID'].astype(str)).tolist()
    this_shift_ids.insert(0, work_id)  # Add the work location ID

    # Extract the distance and duration matrices for the current shift
    this_shift_distances = dist_mat.loc[this_shift_ids, this_shift_ids].copy()
    this_shift_durations = time_mat.loc[this_shift_ids, this_shift_ids].copy()

    # Create a DataFrame of employee IDs
    emps = this_shift_distances.reset_index()[['index']].rename(columns={'index': 'Employee_System_ID'})

    # Read employee details from the Excel file
    emp_details2 = pd.read_excel('Sample-Data.xlsx', sheet_name='Emp Details')

    # Add employee IDs and set the index to employee IDs
    emp_details2['Employee_System_ID'] = 'EMP' + emp_details2['Employee_System_ID'].astype(str)
    emp_details3 = emp_details2.copy()

    # Get the indices of female employees
    emp_details2 = emp_details2[['Employee_System_ID', 'Emp_Gender']]
    emps = emps.merge(emp_details2, on='Employee_System_ID', how='left')
    female_ind = list(emps[emps['Emp_Gender'] == 'Female'].index)

    # Create an instance of the routing class
    route = route_generator.Routing(
        distance_matrix=this_shift_distances,
        duration_matrix=this_shift_durations,
        max_vehicle_capacity=MaxVehicleCapacity,
        genders=genders,
        female_ind=female_ind,
        cost_per_km=cost_per_km,
        cost_per_min=cost_per_min,
        max_distance=MaxDist,
        max_time=MaxTime,
        work_id=work_id)
    
    # Solve the routing problem and get the final route and route distance
    final_route, route_distance = route.solve()

    # Create a mapping between employee IDs and their positions in the final route
    emp_id_map = {}
    for i in range(len(this_shift_distances.reset_index())):
        emp_id_map[i] = this_shift_distances.reset_index()['index'][i]

    # Convert the final route from a nested list to a list of lists
    new_list = [[emp_id_map[index] for index in sublist] for sublist in final_route]

    # Create an empty DataFrame to store the route details
    out = pd.DataFrame()

    # Replace 'EMP' prefix with empty string in employee IDs
    emp_details3['Employee_System_ID'] = emp_details3['Employee_System_ID'].str.replace('EMP', '')
    
    # Iterate through each route
    for i in range(len(new_list)):
        # Reverse the route and route distance if it's a pickup route
        if route_choice == 'pickup':
            new_list[i].reverse()
            route_distance[i].reverse()
            route_distance[i] = route_distance[i][1:]

        # Otherwise, remove the office location from the route distance
        else:
            route_distance[i] = route_distance[i][:-1]

        # Create a temporary DataFrame for the current route
        temp_df = pd.DataFrame()
        temp_df['Employee_System_ID'] = new_list[i][1:-1]
        temp_df['Employee_System_ID'] = temp_df['Employee_System_ID'].astype(str)
        temp_df['Employee_System_ID'] = temp_df['Employee_System_ID'].str.replace('EMP', '')
        temp_df = temp_df.merge(emp_details3, on='Employee_System_ID', how='left')  # Merge employee details
        if route_choice == 'pickup':
            temp_df['Route Type'] = 'Pickup'  # Set route type to 'Pickup'
        else:
            temp_df['Route Type'] = 'Drop'  # Set route type to 'Drop'
        temp_df['cab'] = i + 1  # Assign cab number
        temp_df['route_order'] = list(range(1, len(new_list[i][1:-1]) + 1))  # Assign route order
        temp_df['capacity'] = 4  # Set vehicle capacity
        temp_df['total_route_distance'] = route_distance[i]  # Set total route distance

        # Merge with office location distance data
        office_dist = dist_mat.T.reset_index()[['index', 'OFC000']].rename(columns={'index': 'Employee_System_ID'})
        office_dist['Employee_System_ID'] = office_dist['Employee_System_ID'].str.replace('EMP', '')
        temp_df = temp_df.merge(office_dist, on='Employee_System_ID', how='left').rename(columns={'OFC000': 'distance_to_office'})
        temp_df['distance_to_office'] = temp_df['distance_to_office'] / 1000  # Convert distance to kilometers
        
        # Check for possible swap detection
        if len(temp_df != 0):

            if route_choice == 'pickup':

                # Check if the distance to the office decreases after each pickup
                temp_df['possible_swap_detected'] = 'No' if all(
                    temp_df['distance_to_office'].diff().dropna() <= 0) else 'Yes'

                # Determine if a guard is required based on the gender of the last employee
                is_fem = list(temp_df.iloc[-1:].reset_index(drop=True)['Emp_Gender'])[0]
                temp_df['Guard'] = 'Yes' if is_fem == 'Female' else 'No'

                # Set the shift name
                temp_df['shift'] = f'{shift}'

            else:

                # Check if the distance to the office increases after each drop
                temp_df['possible_swap_detected'] = 'No' if all(
                    temp_df['distance_to_office'].diff().dropna() >= 0) else 'Yes'

                # Determine if a guard is required based on the gender of the first employee
                is_fem = list(temp_df.iloc[0:1].reset_index(drop=True)['Emp_Gender'])[0]
                temp_df['Guard'] = 'Yes' if is_fem == 'Female' else 'No'

                # Set the shift name
                temp_df['shift'] = f'{shift}'

            # Append the temporary DataFrame to the output DataFrame
            out = pd.concat([out, temp_df], ignore_index=True)

        # Set the work location latitude and longitude
        work_latlong = '28.4980845,77.40357'

        # Generate Google Maps URLs for drop routes
        if route_choice == 'pickup':
            base_drop_url = f'https://www.google.co.in/maps/dir/'
            out['MapUrl'] = out.groupby(['shift', 'cab'])['Emp_Drop_LatLong'].transform(
                lambda x: base_drop_url + '/'.join(x) + f'/{work_latlong}'
            )
        else:
            base_drop_url = f'https://www.google.co.in/maps/dir/{work_latlong}/'
            out['MapUrl'] = out.groupby(['shift', 'cab'])['Emp_Drop_LatLong'].transform(
                lambda x: base_drop_url + '/'.join(x)
            )

        # Select the columns to be included in the output DataFrame
        out = out[[
            'shift',
            'Route Type',
            'cab',
            'capacity',
            'route_order',
            'Employee_System_ID',
            'Emp_Name',
            'Emp_Gender',
            'Emp_Location',  # Essential Employee details
            'distance_to_office',
            'total_route_distance',
            'Guard',
            'possible_swap_detected',
            'MapUrl',  # Route stats
            'Emp_Code',
            'Emp_Email',
            'Emp_Address',
            'Emp_NodalPoint',
            'ZoneName',  # Non Essential Details
            'Emp_Pick_LatLong',
            'Emp_Drop_LatLong',
            'Emp_Nodal_LatLong',
            'Work Location'  # Non Essential Details
        ]]

    output = pd.concat([output, out], ignore_index=True)

# Save the output DataFrame to an Excel file
output.to_excel(f'500_employes_{route_choice}.xlsx', index=False)
