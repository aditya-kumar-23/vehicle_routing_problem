from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp
import numpy as np
import pandas as pd
from itertools import combinations


class Routing:
    """
    This class represents a routing problem with multiple vehicles and depots.

    Attributes:
        distance_matrix (numpy.ndarray): The distance matrix between all nodes.
        duration_matrix (numpy.ndarray): The duration matrix between all nodes.
        max_vehicle_capacity (int): The maximum capacity of each vehicle.
        genders (pandas.Series): The genders of all employees.
        female_ind (list): A list of the indices of the female employees.
        cost_per_km (float): The cost per kilometer traveled.
        cost_per_min (float): The cost per minute spent.
        max_distance (float): The maximum distance a vehicle can travel.
        max_time (float): The maximum time a vehicle can spend traveling.
        work_id (int): The index of the work node.
        manager (routing_enums_pb2.RoutingManager): The routing manager object.
        routing (routing_enums_pb2.Routing): The routing object.
        num_vehicles (int): The number of vehicles.
        vehicle_capacities (list): The capacities of each vehicle.

    Methods:
        distance_callback(from_index, to_index): Returns the distance between two nodes.
        duration_callback(from_index, to_index): Returns the duration between two nodes.
        cost_callback(from_index, to_index): Returns the cost between two nodes.
        demand_callback(from_index, to_index): Returns the demand of a node.
        print_solution(solution): Prints the solution to the console.
        get_viable_edges(cost_mat, route_type): Returns a DataFrame of viable edges for a given route type.
        get_route_cum_cost(route, cost_mat): Returns the cumulative cost of a route.
        check_valid_route(nodes): Checks if a route is valid.
        get_route_cost(nodes, cost_mat): Returns the cost of a route.
        initialize_route(female_rule): Initializes the routes.
    """

    def __init__(self, distance_matrix, duration_matrix, max_vehicle_capacity, genders, female_ind,
                 cost_per_km, cost_per_min, max_distance, max_time, work_id) -> None:
        self.dist_df = distance_matrix.copy()
        self.time_df = duration_matrix.copy()
        self.distance_matrix = np.array(distance_matrix).astype(int)
        self.duration_matrix = np.array(duration_matrix).astype(int)

        self.female_ind = female_ind
        self.cost_per_km = cost_per_km
        self.cost_per_min = cost_per_min
        self.max_distance = max_distance
        self.max_time = max_time
        self.max_vehicle_capacity = max_vehicle_capacity
        self.genders = genders
        self.work_id = work_id

        self.depot = 0
        self.max_dist = (self.distance_matrix[0,:] * 1.25).astype(int).tolist()
        self.demands = [0] + np.ones(self.distance_matrix.shape[0]).astype(int).tolist()
        self.cost_matrix = self.distance_matrix / 1000 * self.cost_per_km \
                           + self.duration_matrix / 60 * self.cost_per_min
        self.cost_matrix[self.female_ind, 0] = self.cost_matrix[self.female_ind, 0] + 300

        self.demand_mat = np.tile(self.demands, (len(self.demands), 1)).T.tolist()

        for i in self.female_ind:
            self.demand_mat[i][0] = self.demand_mat[i][0] + 1

    def distance_callback(self, from_index, to_index):
        """
        Returns the distance between the two nodes.

        Args:
            from_index (int): The index of the starting node.
            to_index (int): The index of the ending node.

        Returns:
            int: The distance between the two nodes.
        """

        
    # Convert from routing variable Index to distance matrix NodeIndex.
        from_node = self.manager.IndexToNode(from_index)
        to_node = self.manager.IndexToNode(to_index)
        return self.distance_matrix[from_node][to_node]
    
    def duration_callback(self, from_index, to_index):
        """
        Returns the duration between the two nodes.

        Args:
            from_index (int): The index of the starting node.
            to_index (int): The index of the ending node.

        Returns:
            int: The duration between the two nodes.
        """
        # Convert from routing variable Index to distance matrix NodeIndex.
        from_node = self.manager.IndexToNode(from_index)
        to_node = self.manager.IndexToNode(to_index)
        return self.duration_matrix[from_node][to_node]


    def cost_callback(self, from_index, to_index):
        """
        Returns the cost between the two nodes.

        Args:
            from_index (int): The index of the starting node.
            to_index (int): The index of the ending node.

        Returns:
            int: The cost between the two nodes.
        """
        # Convert from routing variable Index to distance matrix NodeIndex.
        from_node = self.manager.IndexToNode(from_index)
        to_node = self.manager.IndexToNode(to_index)
        return self.cost_matrix[from_node][to_node]


    def demand_callback(self, from_index, to_index):
        """
        Returns the demand of the node.

        Args:
            from_index (int): The index of the starting node.
            to_index (int): The index of the ending node.

        Returns:
            int: The demand of the node.
        """
        # Convert from routing variable Index to demands NodeIndex.
        from_node = self.manager.IndexToNode(from_index)
        to_node = self.manager.IndexToNode(to_index)
        return self.demand_mat[from_node][to_node]


    def print_solution(self, solution):
        """
        Prints the solution to the console.

        Args:
            solution (routing_enums_pb2.Assignment): The solution to the routing problem.
        """
        print(f"Objective: {solution.ObjectiveValue()}")
        total_distance = 0
        total_load = 0
        vehicle_route_list = []
        route_distance_list = []
        for vehicle_id in range(self.num_vehicles):
            vehicle_route = []
            index = self.routing.Start(vehicle_id)
            plan_output = f"Route for vehicle {vehicle_id}:\n"
            route_distance = 0
            route_load = 0
            route_lst = []
            while not self.routing.IsEnd(index):
                node_index = self.manager.IndexToNode(index)
                vehicle_route.append(node_index)
                route_load += self.demands[node_index]
                plan_output += f" {node_index} Load({route_load}) -> "
                previous_index = index
                index = solution.Value(self.routing.NextVar(index))
                route_distance += self.distance_callback(previous_index, index)
                if index != self.work_id:
                    route_lst.append((route_distance) / 1000)
            plan_output += f" {self.manager.IndexToNode(index)} Load({route_load})\n"
            plan_output += f"Distance of the route: {route_distance / 1000:.2f}km\n"
            route_distance_list.append(route_lst)
            plan_output += f"Load of the route: {route_load}\n"
            print(plan_output)
            total_distance += route_distance
            total_load += route_load
            vehicle_route.append(self.manager.IndexToNode(index))
            vehicle_route_list.append(vehicle_route)

        print(f"Total distance of all routes: {total_distance / 1000}km")
        print(f"Total load of all routes: {total_load}")
        return vehicle_route_list, route_distance_list

    def get_viable_edges(self, cost_mat, route_type='drop', DirectDistRatio=1.25):
        """
        Returns a DataFrame of viable edges for a given route type.

        Args:
            cost_mat (pandas.DataFrame): The cost matrix between all nodes.
            route_type (str): The type of route ('pickup' or 'drop').
            DirectDistRatio (float): The direct distance ratio.

        Returns:
            pandas.DataFrame: A DataFrame of viable edges.
        """
        if route_type not in ('pickup', 'drop'):
            raise NotImplementedError
        if route_type == 'pickup':
            return None

        from_work_dist = cost_mat.loc[self.work_id, :].rename('distance')
        work_to_source = cost_mat.apply(lambda x: from_work_dist, axis=0)
        work_to_dest = cost_mat.apply(lambda x: from_work_dist, axis=1)
        viable_edges = (work_to_source + cost_mat <= (DirectDistRatio * work_to_dest).clip(upper=self.max_distance*1000)) \
                    & (work_to_source <= work_to_dest + 2000)

        return viable_edges
    
    def get_route_cum_cost(self, route, cost_mat):
        """
        Calculates the cumulative cost of a route.

        Args:
            route (list): A list of node indices representing the route.
            cost_mat (pandas.DataFrame): The cost matrix between all nodes.

        Returns:
            list: A list of the cumulative costs of each node in the route.
        """
        cost = 0
        pr = None
        cumcost = []
        for r in route:
            if pr:
                cost += cost_mat.loc[pr, r]
            cumcost.append(cost)
            pr = r

        return cumcost
    
    def check_valid_route(self, nodes):
        """
        Checks if a route is valid.

        Args:
            nodes (pandas.Series): A series of node indices representing the route.

        Returns:
            bool: True if the route is valid, False otherwise.
        """
        non_dummy_nodes = nodes[nodes != 'STOP']
        return len(non_dummy_nodes) == non_dummy_nodes.nunique()
    
    def get_route_cost(self, nodes, cost_mat):
        """
        Calculates the cost of a route.

        Args:
            nodes (pandas.Series): A series of node indices representing the route.
            cost_mat (pandas.DataFrame): The cost matrix between all nodes.

        Returns:
            float: The cost of the route.
        """
        cost = 0
        nodes = nodes.dropna()
        for i in range(0, len(nodes)-1):
            cost += cost_mat.loc[nodes[i], nodes[i+1]]

        return cost


    def initialize_route(self, female_rule=True, DirectDistRatio=1.25):
        '''
        Initializes routes for employee transportation.

        Inputs:
        - female_rule: If true, removes routes where women are dropped Nth where N = MaxVehicleCapacity.
        - DirectDistRatio: Ratio used to calculate a threshold for direct distances.

        Returns:
        A list of routes where each route is represented as a list of employee node indices.

        Note: Make sure to set other necessary attributes before calling this method.
        '''
        # Extract employee genders and convert them to uppercase initials
        emp_genders = self.genders.rename('gender').str.slice(0, 1).str.upper()
        female_rule_cap = int(female_rule)
        this_shift_durations = self.time_df.copy()
        this_shift_distances = self.dist_df.copy()
        print(f'Initializing Route with a max of {self.max_vehicle_capacity} passengers per vehicle')
        emp_node_index_map = pd.Series(range(this_shift_durations.shape[0]), index=this_shift_distances.index)

        # Add dummy node for end of trip
        this_shift_distances.loc[:, 'STOP'] = 0
        this_shift_distances.loc[self.work_id, 'STOP'] = self.max_distance * 2 * 1000
        this_shift_distances.loc['STOP', :] = self.max_distance * 2 * 1000
        this_shift_distances.loc['STOP', 'STOP'] = 0
        this_shift_durations.loc[:, 'STOP'] = 0
        this_shift_durations.loc[self.work_id, 'STOP'] = self.max_time * 2 * 60
        this_shift_durations.loc['STOP', :] = self.max_time * 2 * 60
        this_shift_durations.loc['STOP', 'STOP'] = 0

        # Direct distances
        from_work_dist = this_shift_distances.loc[self.work_id, :].rename('distance')
        A = self.get_viable_edges(cost_mat=this_shift_distances)
        valid_connections = pd.DataFrame(np.linalg.matrix_power(A, self.max_vehicle_capacity),
                                          index=A.index, columns=A.columns)
        emp_list = pd.DataFrame(emp_genders).join(from_work_dist, how='inner').sort_values('distance', ascending=False)
        emp_list['remaining'] = True
        routes = []

        # Iterate until all employees have been assigned to a route
        while emp_list.remaining.any():
            remaining_emps = emp_list.index[emp_list.remaining]
            current_emp = remaining_emps[0]
            cap_penalty = female_rule_cap if emp_list.loc[current_emp, 'gender'] == 'F' else 0
            idx = valid_connections.loc[remaining_emps[1:], current_emp]
            valid_nodes = idx[idx].index
            st = self.work_id
            ed = current_emp
            comb_nodes = valid_nodes
            maxR = min(len(comb_nodes), self.max_vehicle_capacity - 1 - cap_penalty)
            minRoute = None
            minCost = self.max_distance * 1000

            # Find the route with the minimum cost
            for r in range(maxR, 0, -1):
                for i, c in enumerate(combinations(comb_nodes, r)):
                    if i >= 1000:
                        break
                    route = list((st, *c[::-1], ed))
                    cum_dist = self.get_route_cum_cost(route=route, cost_mat=this_shift_distances)
                    cum_times = self.get_route_cum_cost(route=route, cost_mat=this_shift_durations)

                    dir_dist = from_work_dist[route]
                    thresh = (dir_dist * DirectDistRatio).clip(upper=self.max_distance * 1000)
                    if all(cum_dist <= thresh) and all(np.array(cum_times) <= self.max_time * 60):
                        if cum_dist[-1] <= minCost:
                            minRoute = route
                            minCost = cum_dist[-1]
                if minRoute:
                    break

            if minRoute is None:
                minRoute = [st, ed]
            route_emp_nodes = emp_node_index_map[minRoute[1:]].tolist()
            routes.append(route_emp_nodes)
            emp_list.loc[minRoute[1:], 'remaining'] = False

        return routes


    def solve(self):
        # Generate initial routes
        self.initial_route_list = self.initialize_route()
        self.num_vehicles = len(self.initial_route_list) + 25
        self.vehicle_capacities = [self.max_vehicle_capacity] * self.num_vehicles

        # Create routing model and manager
        self.manager = pywrapcp.RoutingIndexManager(len(self.distance_matrix), self.num_vehicles, self.depot)
        self.routing = pywrapcp.RoutingModel(self.manager)

        # Register transit callbacks for cost, distance, duration, and demand
        cost_callback_index = self.routing.RegisterTransitCallback(self.cost_callback)
        distance_callback_index = self.routing.RegisterTransitCallback(self.distance_callback)
        duration_callback_index = self.routing.RegisterTransitCallback(self.duration_callback)
        demand_callback_index = self.routing.RegisterTransitCallback(self.demand_callback)

        # Define cost of each arc
        self.routing.SetArcCostEvaluatorOfAllVehicles(cost_callback_index)

        # Add distance and duration dimensions to the model
        dimension_name = "Distance"
        self.routing.AddDimension(
            distance_callback_index,
            0,  # no slack
            self.max_distance * 2 * 1000,  # vehicle maximum travel distance
            True,  # start cumul to zero
            dimension_name,
        )
        distance_dimension = self.routing.GetDimensionOrDie(dimension_name)

        dimension_name = "Duration"
        self.routing.AddDimension(
            duration_callback_index,
            0,  # no slack
            self.max_time * 2 * 60,  # vehicle maximum travel duration
            True,  # start cumul to zero
            dimension_name,
        )
        duration_dimension = self.routing.GetDimensionOrDie(dimension_name)

        # Set drop distance as 1.25 * direct distance
        for location_idx, max_dist in enumerate(self.max_dist):
            if location_idx == self.depot:
                continue
            index = self.manager.NodeToIndex(location_idx)
            max_dist_min = min(self.max_distance * 1000, max_dist)
            distance_dimension.CumulVar(index).SetRange(0, max_dist_min)

        # Add dimension for vehicle capacity
        self.routing.AddDimensionWithVehicleCapacity(
            demand_callback_index,
            0,  # null capacity slack
            self.vehicle_capacities,  # vehicle maximum capacities
            True,  # start cumul to zero
            "Capacity",
        )

        # Close model with custom search parameters
        search_parameters = pywrapcp.DefaultRoutingSearchParameters()
        search_parameters.first_solution_strategy = (
            routing_enums_pb2.FirstSolutionStrategy.AUTOMATIC)
        search_parameters.local_search_metaheuristic = (
            routing_enums_pb2.LocalSearchMetaheuristic.AUTOMATIC)
        search_parameters.time_limit.FromSeconds(100)
        self.routing.CloseModelWithParameters(search_parameters)

        # Get initial solution from routes after closing the model
        initial_solution = self.routing.ReadAssignmentFromRoutes(self.initial_route_list, True)

        # Solve the problem
        solution = self.routing.SolveFromAssignmentWithParameters(
            initial_solution, search_parameters
        )

        print(f'Routing Status: {self.routing.status()}')

        # Print and return the final solution
        vrlist, route_dist = self.print_solution(solution)
        return vrlist, route_dist


