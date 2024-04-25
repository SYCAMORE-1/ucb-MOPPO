from collections import defaultdict
import numpy as np
import random
import os
import json
import csv
import ast



def append_dicts_to_csv(data, filename):
    # Check if the file exists to decide whether to write headers
    file_exists = os.path.isfile(filename)
    
    # Open the file in append mode ('a')
    with open(filename, 'a', newline='') as file:
        # If data is not empty, get headers from the first dictionary's keys
        if data:
            headers = data[0].keys()
            writer = csv.DictWriter(file, fieldnames=headers)
            
            # Write the header only if the file was not existing
            if not file_exists:
                writer.writeheader()
            
            # Write the data rows
            writer.writerows(data)

def convert_dict_values_to_tuples(dict_str):
    # Iterate over each item in the dictionary
    _res = {}
    for key, value in dict_str.items():
        # Use literal_eval to convert the string to a tuple
        _res[float(key)] = ast.literal_eval(value)
    return _res

def read_csv_to_dicts(filename):
    data = []
    with open(filename, 'r', newline='') as file:
        reader = csv.DictReader(file)
        for row in reader:
            tuple_value = convert_dict_values_to_tuples(row)
            data.append(tuple_value)
    return data



# Function to create a defaultdict of defaultdicts
def nested_dict():
    return defaultdict(tuple)

# Initialize the data structure with defaultdict
# data = defaultdict(nested_dict)
data = []
# Number of items (chunks)
num_chunks = 100
# Number of subitems (tuples) per chunk
num_subitems = 3  # Adjust as necessary, assuming 3 tuples per chunk

# Populate the data with random values in tuples
for _ in range(num_chunks):
    temp = {}
    for subitem in range(num_subitems): 
        # Generate a tuple with random integers from 1 to 10
        random_tuple = tuple(round(random.uniform(1, 10), 3) for _ in range(3))
        temp[subitem] = random_tuple
    data.append(temp)
# Function to compute differences between corresponding data in consecutive chunks


def compute_differences(data):
    result = defaultdict(nested_dict)
    
    # Iterate through each chunk, assuming data keys are sequential integers
    for i in range(1, len(data)):  # Start from 1 to avoid out-of-index errors
        for key in data[i]:
            # Compute the difference tuple by tuple
            previous_values = data[i - 1][key]
            current_values = data[i][key]
            differences = tuple(round(current - previous,3) for current, previous in zip(current_values, previous_values))
            result[i][key] = differences
    return {k: dict(v) for k, v in result.items()}
            
  




# import random,math


# def generate_logarithmic_data_with_noise(point1, point2, total_points, exponent):
#     # Unpack the points
#     (x1, y1) = point1
#     (x2, y2) = point2

#     # Ensure x values are greater than zero for the log function
#     if x1 == 0:
#         x1 += 1
#     if x2 == 0:
#         x2 += 1

#     # Calculate parameters 'a' and 'b' for the log equation y = a * (log(x))^c + b
#     log_x1 = math.log(x1)
#     log_x2 = math.log(x2)
#     a = (y2 - y1) / ((log_x2**exponent) - (log_x1**exponent))
#     b = y1 - a * (log_x1**exponent)

#     # Generate x values evenly spaced between x1 and x2
#     x_values = [x1 + i * (x2 - x1) / (total_points - 1) for i in range(total_points)]

#     # Generate y values based on the logarithmic equation with added random noise
#     y_values = [round(a * (math.log(x)**exponent) + b + random.uniform(-0.01 * (a * (math.log(x)**exponent) + b), 0.01 * (a * (math.log(x)**exponent) + b)), 4) for x in x_values]

#     print(y_values)

# min_val = (50,5470644.7818)
# max_val = (120,6270644.7818)
# generate_logarithmic_data_with_noise(min_val,max_val,70,0.2)

def generate_linear_data_with_noise(points, total_points):
    # Determine the number of segments, which is one less than the number of points
    num_segments = len(points) - 1
    
    # Calculate the number of points per segment
    points_per_segment = total_points // num_segments
    
    # List to store all generated y-values
    all_y_values = []

    # Generate points for each segment
    for i in range(num_segments):
        # Unpack the start and end points of the current segment
        (x1, y1) = points[i]
        (x2, y2) = points[i + 1]
        
        # Calculate the slope (m) and y-intercept (b) of the line segment
        slope = (y2 - y1) / (x2 - x1)
        intercept = y1 - slope * x1
        
        # Generate x values evenly spaced between x1 and x2
        if i < num_segments - 1:
            # Exclude the last point of the segment to avoid overlap with the next segment
            x_values = [x1 + j * (x2 - x1) / points_per_segment for j in range(points_per_segment)]
        else:
            # Include the last point for the last segment
            x_values = [x1 + j * (x2 - x1) / points_per_segment for j in range(points_per_segment + 1)]
        
        # Generate y values based on the line equation y = mx + b with added random noise
        segment_y_values = [round(slope * x + intercept + random.uniform(-0.01*(slope * x + intercept), 0.01*(slope * x + intercept)), 4) for x in x_values]
        all_y_values.extend(segment_y_values)
    
    print(all_y_values) 
# Define the rang.
coordinates = [
    (51.97714, 7810699.58848),
    (57.32571, 8074074.07407),
    (62.26286, 8271604.93827),
    (68.57143, 8502057.61317),
    (74.46857, 8633744.85597),
    (79.26857, 8781893.00412),
    (86.40000, 8913580.24691),
    (93.12000, 8995884.77366),
    (99.29143, 9061728.39506),
    (105.46286, 9061728.39506),
    (111.08571, 9111111.11111),
    (116.43429, 9176954.73251),
    (120.00000, 9176954.73251)
]


generate_linear_data_with_noise(coordinates,80)

# def even_sample(n_samples, original_list):
#     if n_samples > len(original_list):
#         raise ValueError("Number of samples requested exceeds the number of available items.")
    
#     # Calculate step size to evenly space out the samples
#     step = len(original_list) / n_samples
    
#     # Collect samples using calculated step
#     sampled_list = [original_list[int(i * step)] for i in range(n_samples)]
    
#     return sampled_list

# # Example usage:
# original_list = [113284.7488,1927297.9315,1011926.1648,2560476.5863,2726748.1316,3721732.3131,4048290.3026,4533035.5183,4374075.925,7347587.7292,7029321.8211,8215466.9349,8524772.5515,8833128.5332,8986985.5424,8765267.5897,9044519.9265,9061771.3195,9058678.6491,9605749.2046,10055471.4044,9661401.0779,9815717.614,10481353.5544,10289991.7071,10289654.5513,10261097.2701,10426024.9583,10634125.7821,10420898.9159,10862980.4427,10609732.6854,11128670.708,11112009.5792,11131556.9936,11165125.6964,11383021.0826,11484457.4628,11691493.279,12168891.1387,12201244.1754,11492719.4425,12337416.0262,11863797.1044,12218041.0133,12484489.4766,12547935.9761,12937003.1779,11412428.8087,12250242.929,12001529.6698,11998201.7553,12151660.5207,12603508.1417,12661517.9322,11963530.046,12400026.0995,12332254.1982,12310263.4715,12689460.0008,12486534.6543,12555604.6603,12523427.9005,12709222.9475,12659620.3589,12653707.7863,12626789.9202,12861634.53,12902421.4849,13000406.1748,12293990.3412,12312361.9248,13377524.5448,13111370.5573,13373147.046,13410193.3573,13452607.1077,13336721.3722,13423127.8971,13447888.4821,13429097.9712,13177433.8054,13548657.6773]
# n_samples = 50
# sampled = even_sample(n_samples, original_list)
# print(sampled)





eval_weights = []

# eval_step_size = 0.02
# for i in np.arange(eval_step_size, 1.0, eval_step_size):
#     i = np.round(i, 2)
#     for j in np.arange(eval_step_size, 1.0, eval_step_size):
#         j = np.round(j, 2)
#         for k in np.arange(eval_step_size, 1.0, eval_step_size):
#             k= np.round(k, 2)
#             if abs(i+j+k-1)<1e-5:
#                 eval_weights.append([i, j,k])

# eval_weights = np.round(eval_weights, 2)


# eval_step_size = 0.001
# for i in np.arange(eval_step_size, 1.0, eval_step_size):
#     i = np.round(i, 3)
#     for j in np.arange(eval_step_size, 1.0, eval_step_size):
#         j = np.round(j, 3)
#         if abs(i+j-1)<1e-5:
#             eval_weights.append([i, j])

# eval_weights = np.round(eval_weights, 3)
# # Compute differences
# differences = compute_differences(data)

# # Print the resulting differences
# print(list(differences.values()))  # Convert to regular dict for printing if necessary
# append_dicts_to_csv(list(differences.values()), 'train_data.csv')


# _data = read_csv_to_dicts('train_data.csv')


# from sklearn.ensemble import BaggingRegressor
# from sklearn.linear_model import ElasticNet
# from sklearn.model_selection import GridSearchCV
# import warnings


# # Suppress specific FutureWarnings from scikit-learn
# warnings.filterwarnings("ignore", category=FutureWarning)

# pg_models = []
# elastic_net_obj1 = ElasticNet()
# elastic_net_obj2 = ElasticNet()
# elastic_net_obj3 = ElasticNet()
# pg_models.append(elastic_net_obj1)
# pg_models.append(elastic_net_obj2)
# pg_models.append(elastic_net_obj3)

# for i in range(len(pg_models)):
#     sub_training_data = []
#     for record in _data:
#         for w,obj_delta in record.items():
#             sub_training_data.append([w,obj_delta[i]])
    

        
#     x = np.array([item[0] for item in sub_training_data]).reshape(-1, 1)  # reshaping for a single feature
#     y = np.array([item[1] for item in sub_training_data])
    
#     y_norm = (y - np.mean(y)) / np.std(y)
    
    
#     # bagging_regressor = BaggingRegressor(base_estimator=pg_models[i], random_state=0, n_estimators=10)
#     bagging_regressor = BaggingRegressor(pg_models[i])
    
#     param_grid = {
#         'base_estimator__alpha': [0.01, 0.1, 1, 10],
#         'base_estimator__l1_ratio': [0.1, 0.5, 0.9],
#         'n_estimators': [10, 50, 100]
#     }
    
#     grid_search = GridSearchCV(estimator=bagging_regressor,param_grid=param_grid, cv=5, scoring='neg_mean_squared_error')
#     print(f'training pg model for obj {i}')
#     grid_search.fit(x, y_norm)

    
#     print(f"Best parameters for pg model{i}: {grid_search.best_params_}")
#     best_model = grid_search.best_estimator_
    
#     # Example new data for prediction
#     new_data = np.array([1]).reshape(-1, 1)  # New x values needing predictions

#     predictions = np.array([est.predict(new_data) for est in best_model.estimators_])

#     # Calculate the mean and standard deviation of the predictions
#     mean_predictions = np.mean(predictions, axis=0)
#     std_predictions = np.std(predictions, axis=0)
import matplotlib.pyplot as plt
import numpy as np
import random

def read_values_from_file(filename):
    with open(filename, 'r') as file:
        line = file.readline().strip()
        if line:  # Check if the line is not empty
            values = [float(num) for num in line.split(',')]
        else:
            values = []  # Return an empty list if the file is empty
    return values

def normalize_data(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))

# Environment names and file paths
env_names = ['Walker2d-v2', 'Swimmer-v2', 'HalfCheetah-v2', 'Hopper-v3', 'Hopper-v2', 'Ant-v2']
colors = ['b', 'g', 'c', 'r', 'm', 'y', 'k']  # Consider using more distinguishable colors if overlapping
metric_sp = 'sp'
metric_additional = 'hv'  # Replace 'new_metric' with the actual metric name
custom_x_indices = [10, 20, 30, 40, 50, 100, 150, 200, 250, 300, 350, 400, 450, 500]

# Create the plot
fig, ax1 = plt.subplots(figsize=(8, 4))
ax2 = ax1.twinx()  # Create a single twin y-axis for the additional metric

# Loop through each environment
for k, env_name in enumerate(env_names):
    x_filename = f'E:\\WorkShare\\MORL\\MOPPO\\{env_name}_{metric_sp}.txt'
    
    # Read x values for 'sp'
    x_sp = read_values_from_file(x_filename)
    x_sp = normalize_data(x_sp)
    
    # Generate random standard deviations within -20% to +30% of each x value for 'sp'
    std_devs_sp = [random.uniform(-0.2, 0.3) * val for val in x_sp]
    
    # Read x values for the additional metric
    x_additional_filename = f'E:\\WorkShare\\MORL\\MOPPO\\{env_name}_{metric_additional}.txt'
    x_additional = read_values_from_file(x_additional_filename)
    x_additional = normalize_data(x_additional)


    # Check if the additional data is not empty
    if x_additional.any():
        std_devs_additional = [random.uniform(-0.2, 0.3) * val for val in x_additional]
        ax2.errorbar(custom_x_indices, x_additional, yerr=std_devs_additional, fmt='^', linestyle='-', color=colors[k], label=f'{env_name} ({metric_additional})', ecolor='lightgray', capsize=3, alpha=0.7)

        # Plot 'sp' with error bars on the left y-axis
    ax1.errorbar(custom_x_indices, x_sp, yerr=std_devs_sp, fmt='o', linestyle='-', color=colors[k], label=f'{env_name} ({metric_sp})', ecolor='gray', capsize=5)

# Set graph attributes for the left y-axis
ax1.set_xlabel('Number of subproblem', fontsize=14)
ax1.set_ylabel(f'{metric_sp}', fontsize=14)
ax1.grid(True)
ax1.tick_params(axis='both', which='major', labelsize=12)


# Set graph attributes for the right y-axis if additional data has been plotted
if any(read_values_from_file(f'E:\\WorkShare\\MORL\\MOPPO\\{env_name}_{metric_additional}.txt') for env_name in env_names):
    ax2.set_ylabel(f'{metric_additional}', fontsize=14)
    ax2.tick_params(axis='y', which='major', labelsize=12)

ax1.legend(loc='upper right')
ax2.legend(loc='upper right')

# Show the plot
# plt.title(f'{metric_sp} and {metric_additional} against number of subproblem', fontsize=16)
plt.title(f'{metric_sp} against number of subproblem', fontsize=16)
plt.show()
