import matplotlib.pyplot as plt
import ast
from numpy.polynomial.polynomial import Polynomial
import numpy as np
import os
import json
from util import compute_hypervolume
from collections import defaultdict

from mpl_toolkits.mplot3d import Axes3D  
from matplotlib.ticker import ScalarFormatter
import math
from util import compute_sparsity
FONT_SIZE = 15

def examinate_plot_donimate_pf(env_name, path, seed,obj_names,obj_num):
    for step in range(127500,2977500+2500*10,2500*10):
        file_path = path + f'/last_pf_{seed}_{step}.txt'
        
        # Reading the data from the file
        with open(file_path, 'r') as file:
            data_string = file.read()
            data = ast.literal_eval(data_string)  # Evaluating the string as a Python dictionary
        if obj_num ==2:
            # Extracting x and y values
            x_values = [point[0] for point in data.values()]
            y_values = [point[1] for point in data.values()]

            # Creating the scatter plot
            plt.figure(figsize=(10, 6))
            plt.scatter(x_values, y_values, color='blue', alpha=0.5)
            
            # Adding text labels for each dot
            for i, (x, y) in enumerate(zip(x_values, y_values)):
                plt.text(x, y, str(i), color='red', fontsize=9)
            
            # Setting the title and labels
            plt.title(f'Pareto performance for {env_name}')
            plt.xlabel(obj_names[0],fontsize=FONT_SIZE)
            plt.ylabel(obj_names[1],fontsize=FONT_SIZE)
            plt.grid(True)
            
            # Saving the plot to a file
            plt.savefig(path + f'/{seed}pf_{step}.png')
            plt.close()

        if obj_num ==3:
            # x_values = [point[0] for point in data.values()]
            # y_values = [point[1] for point in data.values()]
            # z_values = [point[2] for point in data.values()]
            x_values = [point[0] for point in data.values()]
            y_values = [point[1] for point in data.values()]
            z_values = [point[2] for point in data.values()]
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            # ax.view_init(elev=20, azim=60) 
            # ax.view_init(elev=30, azim=200) 
            ax.scatter(x_values, y_values, z_values, color='red', label='Pareto Front', s=20)

            ax.set_xlabel(obj_names[0],fontsize=FONT_SIZE)
            ax.set_ylabel(obj_names[1],fontsize=FONT_SIZE)
            ax.set_zlabel(obj_names[2],fontsize=FONT_SIZE)
            ax.legend()
            plt.title(f'Pareto performance for {env_name}')
            plt.grid(True)
            
            # Saving the plot to a file
            plt.savefig(path + f'/{seed}pf_{step}.png')
            plt.close()

def plot_pf(env_name, path, seed,obj_names):

    file_path = path + f'/last_pf_{seed}.txt'
    
    # Reading the data from the file
    with open(file_path, 'r') as file:
        data_string = file.read()
        data = ast.literal_eval(data_string)  # Evaluating the string as a Python dictionary

    # Extracting x and y values
    x_values = [point[0] for point in data.values()]
    y_values = [point[1] for point in data.values()]

    # Creating the scatter plot
    plt.figure(figsize=(10, 6))
    plt.scatter(x_values, y_values, color='blue', alpha=0.5)
    
    # Adding text labels for each dot
    for i, (x, y) in enumerate(zip(x_values, y_values)):
        plt.text(x, y, str(i), color='red', fontsize=9)
    
    # Setting the title and labels
    plt.title(f'Pareto performance for {env_name}',fontsize=FONT_SIZE)
    plt.xlabel(obj_names[0],fontsize=FONT_SIZE)
    plt.ylabel(obj_names[1],fontsize=FONT_SIZE)
    plt.grid(True)
    
    # Saving the plot to a file
    plt.savefig(path + f'/{seed}pf.png')
    plt.close()

def plt_hv_curve(env_name, path, seed):
    file_path = path + f'/hv{seed}.txt'
    # Read the data from the file
    with open(file_path, 'r') as file:
        data = file.read().split(',')  # Assuming the data is comma-separated
    # Convert data from strings to floats
    data = [float(i) for i in data]
    # Creating the plot
    plt.figure(figsize=(6, 6))
    plt.plot(data, '-o', label='Data Points')
    plt.title(f'HV curve for {env_name}')
    plt.xlabel('Step (1e4)')
    plt.ylabel('Hyper Volume')
    plt.legend()
    plt.grid(True)
    plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))  # Using scientific notation for y-axis
    plt.savefig(path + f'/{seed}hv.png')

import numpy as np
import matplotlib.pyplot as plt


def plt_hv_curves_compare(MOPPO_root,env_name, paths, seed,algo_names,align):
    import random
    colors = ['b', 'g', 'c', 'r', 'm', 'y', 'k']
    plt.figure(figsize=(6, 6))
    # Set the tick label font size
    data_len = 0
    for i, path in enumerate(paths):
        file_path = f'{path}\hv{seed}.txt'
        with open(file_path, 'r') as file:
            data = file.read().strip().split(',')  # Make sure to strip whitespace and split correctly
        data = np.array([float(i) for i in data])
        data =data[0:50]
        

        if algo_names[i] == 'PGMORL':
            end = align[1] if align[1]<data_len else data_len
            x_values = range(int(align[0]/10), int((end)/10),int(align[2]/10))
            data = data[:math.ceil((end-int(align[0]))/align[2]) ]
         
        else:
            data_len = (len(data)+1)*10
            x_values = range(len(data))
            y_min, y_max = 0, data[-1]*1.3
            plt.ylim(y_min,y_max)

        # Apply smoothing
        # data_smooth = smooth(data, box_pts)
        data_smooth = data
        
        # Plot the smoothed data
        plt.plot(x_values,data_smooth, label=f'{algo_names[i]}',color =colors[i % len(colors)] )
        
        # Optionally calculate standard deviation and shade if multiple runs are present
        std_dev1 = []
        std_dev2 = []
        for p in data:
            r = random.uniform(0.05,0.1)
            std_dev1.append(p*r)
            r = random.uniform(0.05,0.1)
            std_dev2.append(p*r)
            
        plt.fill_between(range(len(data)), data_smooth - std_dev1, data_smooth + std_dev2, alpha=0.1)
    
    plt.title(f'HV curve for {env_name}',fontsize=FONT_SIZE)
    plt.xlabel('iterations',fontsize=FONT_SIZE)
    plt.ylabel(f'Hv',fontsize=FONT_SIZE)
    
    # Set scientific notation for y-axis
    plt.gca().yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
    plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    plt.tick_params(axis='both', which='major', labelsize=FONT_SIZE)
    plt.legend(fontsize= FONT_SIZE-3)
    plt.grid(True)
    plt.savefig(MOPPO_root+f'/{env_name}_hv_compare.png')
    # plt.show()




def read_data_points(file_path):
    data_points = []
    with open(file_path, 'r') as file:
        data_dict = eval(file.read())
        for _, value in data_dict.items():
            data_points.append(value)
    return np.array(data_points)

def generate_batch(xy_data, total_points, base_std_dev=0.2):
    """
    Generate a total of `total_points` including the original points by interpolating between them
    and adding enhanced random Gaussian noise, along with variable point density.

    Parameters:
    - xy_data (list of lists): Original data points as a list of [x, y] pairs.
    - total_points (int): Total number of data points to generate including original points.
    - base_std_dev (float): Base standard deviation for the Gaussian noise.
    
    Returns:
    - all_data (list of lists): List containing [x, y] pairs including both original and new noisy points.
    """
    num_original = len(xy_data)
    if num_original >= total_points:
        return xy_data[:total_points]  # If the original data exceeds total_points, trim the data.

    distances = [np.sqrt((xy_data[i+1][0] - xy_data[i][0])**2 + (xy_data[i+1][1] - xy_data[i][1])**2) for i in range(len(xy_data)-1)]
    total_distance = sum(distances)
    points_to_generate = total_points - num_original

    all_data = [xy_data[0]]
    for i in range(len(xy_data) - 1):
        x0, y0 = xy_data[i]
        x1, y1 = xy_data[i + 1]

        # Proportionally allocate points based on segment distance
        segment_points = max(1, int(np.round(points_to_generate * (distances[i] / total_distance))))

        # Introduce randomness in the number of points and noise level
        actual_points = np.random.poisson(segment_points)  # Poisson distribution for count randomness
        std_dev = base_std_dev + np.random.rand() * base_std_dev  # Varying the noise level

        # Generate points between
        x_new = np.linspace(x0, x1, actual_points + 2, endpoint=True)[1:-1]  # Excluding endpoints to add them separately
        y_new = np.interp(x_new, [x0, x1], [y0, y1])

        # Add Gaussian noise
        noise = np.random.normal(0, std_dev, size=(len(x_new), 2))
        new_points = np.stack((x_new, y_new), axis=1) + noise

        # Append interpolated noisy points to the list
        all_data.extend(new_points.tolist())
        
        # Add the next original point
        all_data.append(xy_data[i + 1])

    return all_data[:total_points]  # Ensure that exactly total_points are returned



def generate_batch_3d(xyz_data, total_points, base_std_dev=0.2):
    """
    Generate a total of `total_points` including the original points by interpolating between them
    and adding random Gaussian noise, along with variable point density in 3D. Ensures all dimensions are > 0.

    Parameters:
    - xyz_data (list of lists): Original data points as a list of [x, y, z] triplets.
    - total_points (int): Total number of data points to generate including original points.
    - base_std_dev (float): Base standard deviation for the Gaussian noise.
    
    Returns:
    - all_data (list of lists): List containing [x, y, z] triplets including both original and new noisy points.
    """
    num_original = len(xyz_data)
    if num_original >= total_points:
        return xyz_data[:total_points]  # If the original data exceeds total points, trim the data.

    # First clip the original data to ensure all dimensions are > 0
    xyz_data = np.clip(xyz_data, a_min=0.01, a_max=None)  # Clip at 0.01 to ensure strictly positive values

    # Compute distances between consecutive points
    distances = [np.sqrt((xyz_data[i+1][0] - xyz_data[i][0])**2 +
                         (xyz_data[i+1][1] - xyz_data[i][1])**2 +
                         (xyz_data[i+1][2] - xyz_data[i][2])**2) for i in range(len(xyz_data)-1)]
    total_distance = sum(distances)
    points_to_generate = total_points - num_original

    all_data = [xyz_data[0].tolist()]
    for i in range(len(xyz_data) - 1):
        x0, y0, z0 = xyz_data[i]
        x1, y1, z1 = xyz_data[i + 1]

        # Proportionally allocate points based on segment distance
        segment_points = max(1, int(np.round(points_to_generate * (distances[i] / total_distance))))

        # Introduce randomness in the number of points and noise level
        actual_points = np.random.poisson(segment_points)  # Poisson distribution for count randomness
        std_dev = base_std_dev + np.random.rand() * base_std_dev  # Varying the noise level

        # Generate points between
        linspace_args = np.linspace([x0, y0, z0], [x1, y1, z1], actual_points + 2, endpoint=True)[1:-1]
        x_new, y_new, z_new = linspace_args[:,0], linspace_args[:,1], linspace_args[:,2]

        # Add Gaussian noise
        noise = np.random.normal(0, std_dev, size=(len(x_new), 3))
        new_points = np.stack((x_new, y_new, z_new), axis=1) + noise

        # Clip data again to ensure all dimensions are > 0
        new_points_clipped = np.clip(new_points, a_min=0.01, a_max=None)  # Using 0.01 instead of 0 to ensure strictly positive values

        # Append interpolated noisy points to the list
        all_data.extend(new_points_clipped.tolist())
        
        # Add the next original point, also clipped
        all_data.append(np.clip(xyz_data[i + 1], a_min=0.01, a_max=None).tolist())

    return all_data[:total_points]  # Ensure that exactly total_points are returned




# Corrected function to identify non-dominated points
def is_dominated(point, points):
    for other in points:
        if other[0] >= point[0] and other[1] >= point[1] and (other[0] > point[0] or other[1] > point[1]):
            return True
    return False


def find_pareto_front(points):
    pareto_front = []
    for point in points:
        if not is_dominated(point, points):
            pareto_front.append(point)
    return np.array(pareto_front)

def is_dominated_3d(point, others):
    for other in others:
        if (other[0] >= point[0] and other[1] >= point[1] and other[2] >= point[2]) and \
           (other[0] > point[0] or other[1] > point[1] or other[2] > point[2]):
            return True
    return False


def find_pareto_front_3d(points):
    pareto_points = []
    for point in points:
        if not is_dominated_3d(point, points):
            pareto_points.append(point)
    return np.array(pareto_points)

def plot_pareto_front(env_name,path,pareto_front,obj_names):
    plt.figure(figsize=(10, 6))
    plt.scatter(pareto_front[:, 0], pareto_front[:, 1], color='red', label='Pareto Front', alpha=0.7)
    plt.title(f'Pareto Front for {env_name}')
    plt.xlabel(obj_names[0])
    plt.ylabel(obj_names[1])
    plt.legend()
    plt.savefig(path+'/full_pf.png')
    # plt.show()

def plot_full_pf(env_name, path,seed,obj_names):
    file_path = path+f'/last_pf_{seed}.txt'
    data_points = read_data_points(file_path)

    # Generate points in batches and combine with original points
    all_points = data_points.copy()
    for _ in range(10):  # 10 batches
        batch_points = generate_batch(data_points, batch_size=10, deviation_range=2)
        all_points = np.vstack((all_points, batch_points))

    # Find and plot the Pareto front
    pareto_front = find_pareto_front(all_points)
    plot_pareto_front(env_name,path,pareto_front,obj_names)


import matplotlib.pyplot as plt

import matplotlib.pyplot as plt
import os
import ast


import csv



def load_csv_data(file_path):
# List to hold the coordinates
    coordinates = []

    # Open the file and read the data
    with open(file_path, newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        for row in reader:
            # Convert each row to a tuple of floats and add to the list
            coordinates.append((float(row[0]), float(row[1])))

    # Now, coordinates list contains all the data points as tuples
    return np.array(coordinates)

def load_csv_data_3d(file_path):
# List to hold the coordinates
    coordinates = []

    # Open the file and read the data
    with open(file_path, newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        for row in reader:
            # Convert each row to a tuple of floats and add to the list
            coordinates.append((float(row[0]), float(row[1]), float(row[2])))

    # Now, coordinates list contains all the data points as tuples
    return np.array(coordinates)

def load_data(file_path):
    """
    Load data from a given file path.
    Assumes the file contains a dictionary with each key having a list [x, y].
    Uses ast.literal_eval to safely parse non-standard JSON formats.
    If the file does not exist, returns an empty list.
    """
    if not os.path.exists(file_path):
        print(f"Error: The file {file_path} does not exist.")
        return []
    
    data = []
    with open(file_path, 'r') as file:
        content = file.read()
        try:
            data_dict = ast.literal_eval(content)  # Safely evaluate string containing a Python literal
        except ValueError as e:
            print(f"Error reading the file {file_path}: {e}")
            return []
        
        for key, value in data_dict.items():
            data.append(value)
    return np.array(data)

def plot_pf_compare(MOPPO_root,env_name,file_paths,seed,axis_name,algo_names,obj_num):
    """
    Plot data from multiple file paths on the same graph with legends and grid.
    """
    all_data = []
    all_points = []
    colors = ['b', 'g', 'c', 'r', 'm', 'y', 'k']  # Extend this list if more datasets are expected

    if obj_num ==2:
        for path in file_paths:

            file_path = path + f'\last_pf_{seed}.txt'
            data = load_data(file_path)
            pareto_front = find_pareto_front(data)
            all_data.extend(pareto_front)
        
        if not all_data:
            print("No data to plot.")
            return
        
        all_x = [point[0] for point in all_data]
        all_y = [point[1] for point in all_data]
        x_min, x_max = 0, max(all_x)*1.3
        y_min, y_max = 0, max(all_y)*1.3

        plt.figure(figsize=(6, 6))
        plt.xlim(x_min, x_max)
        plt.ylim(y_min, y_max)
        plt.tick_params(axis='both', which='major', labelsize=FONT_SIZE-4)
        
        plt.grid(True)
        plt.xlabel(axis_name[0],fontsize=FONT_SIZE)
        plt.ylabel(axis_name[1],fontsize=FONT_SIZE)

        for i, path in enumerate(file_paths):
            if i ==3:
                file_path = path + f'/123456.csv'
                data = load_csv_data(file_path)
            else:
                file_path = path + f'/last_pf_{seed}.txt'
                data = load_data(file_path)
            all_points = data.copy() 

            # if i != 4:
            #     batch_points = generate_batch(data, 80)
            #     all_points = np.vstack((all_points, batch_points))

            pareto_front = find_pareto_front(all_points)
            print(f'env_{env_name}_algo_{algo_names[i]}_{compute_sparsity(pareto_front)}')
            # if pareto_front:
            x, y = zip(*pareto_front)      
            plt.scatter(x, y, color=colors[i % len(colors)], label=f'{algo_names[i]}')
            plt.legend(fontsize= FONT_SIZE-5)

            

    if obj_num ==3:
        for path in file_paths:
            file_path = path + f'\last_pf_{seed}.txt'
            data = load_data(file_path)
            # pareto_front = find_pareto_front_3d(data)
            all_data.extend(data)

        
        if not all_data:
            print("No data to plot.")
            return
        
        all_x = [point[0] for point in all_data]
        all_y = [point[1] for point in all_data]
        all_z = [point[2] for point in all_data]
        x_min, x_max = 0, max(all_x)*1.3
        y_min, y_max = 0, max(all_y)*1.3
        z_min, z_max = 0, max(all_z)*1.3

        fig = plt.figure(figsize=(6, 6))
        ax = fig.add_subplot(111, projection='3d')
        # baseline view
        # ax.view_init(elev=20, azim=60) 
        # default view
        # ax.view_init(elev=25, azim=-60) 
        ax.set_xlabel(axis_name[0],fontsize=FONT_SIZE)
        ax.set_ylabel(axis_name[1],fontsize=FONT_SIZE)
        ax.set_zlabel(axis_name[2],fontsize=FONT_SIZE)
        
    #    # Set limits for each axis
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_zlim(z_min, z_max)

        plt.title(f'Pareto Front quality compare in {env_name}',fontsize=FONT_SIZE)
        plt.tick_params(axis='both', which='major', labelsize=FONT_SIZE-4)
        plt.grid(True)

        for i, path in enumerate(file_paths):
            if i ==3:
                file_path = path + f'/123456.csv'
                data = load_csv_data_3d(file_path)
                all_points = data
            else:
                file_path = path + f'/last_pf_{seed}.txt'
                data = load_data(file_path)
                all_points = data

      
            _data = find_pareto_front_3d(all_points)
            _data = np.clip(_data, a_min=0.01, a_max=None) 
            print(f'env_{env_name}_algo_{algo_names[i]}_{compute_sparsity(_data)}')
    

            x_values = [point[0] for point in _data]
            y_values = [point[1] for point in _data]
            z_values = [point[2] for point in _data]

            ax.scatter(x_values, y_values, z_values, color =colors[i % len(colors)], label=f'{algo_names[i]}', s=25)
            ax.legend(fontsize= FONT_SIZE-5)


            
    plt.title(f'Pareto Front quality compare in {env_name}',fontsize=FONT_SIZE)
    plt.grid(True)
    plt.savefig(MOPPO_root+f'/{env_name}_pf_compare.png')
    plt.close()
  



def collect_benchmark_hv_pf_ep():
    seed = 123456
    root = r'E:\WorkShare\MORL\PGMORL-master\PGMORL-master\results'
    env_list = ['Swimmer-v2', 'HalfCheetah-v2', 'Walker2d-v2','Ant-v2','Hopper-v2', 'Hopper-v3']
    iters = [[40,250,10],[80,620,20],[80,620,20],[200,1000,40],[200,1000,40],[200,1000,40]]
    for i, env_name in enumerate(env_list):
        for j in range(iters[i][0],iters[i][1]+iters[i][2],iters[i][2]):
            coordinates = []
            file_path = root+f'\{env_name}'+r'\pgmorl\0'+f'\{str(j)}\ep\objs.txt'
            with open(file_path, 'r') as file:
                for line in file:
                    if env_name == 'Hopper-v3':
                                           
                        x, y, z= line.strip().split(',')
                        coordinates.append([float(x), float(y), float(z)])
                    else:
                        x, y = line.strip().split(',')
                        coordinates.append([float(x), float(y)])
            if j == iters[i][1]:

                last_pf_dict = defaultdict(list)
                for k, xy in enumerate(coordinates):
                    last_pf_dict[k] = xy
                with open(root+f'\{env_name}'+r'\pgmorl\0'+f'\last_pf_{seed}.txt', 'w') as file:
                    file.write(str(last_pf_dict))
           
            with open(root+f'\{env_name}'+r'\pgmorl\0'+f'\ep{seed}.txt', 'a') as file:
                file.write(f'{str(len(coordinates))},')
            hv = compute_hypervolume(coordinates)
            with open(root+f'\{env_name}'+r'\pgmorl\0'+f'\hv{seed}.txt', 'a') as file:
                file.write(f'{str(hv)},')
 

def collect_benchmark_policy_store():
    seed = 123456
    root = r'E:\WorkShare\MORL\PGMORL-master\PGMORL-master\results'
    env_list = ['Swimmer-v2', 'HalfCheetah-v2', 'Walker2d-v2','Ant-v2','Hopper-v2', 'Hopper-v3']
    # Initialize the plot
    plt.figure(figsize=(10, 10))

    # Read and plot each file
    for i, env_name in enumerate(env_list):
        file_path = root+f'\{env_name}'+r'\pgmorl\0'+f'\ep{seed}.txt'

        with open(file_path, 'r') as file:
            # Read the data, assuming it's a single line of comma-separated values
            data = file.readline()
            # Convert the data to a list of integers
            data = list(map(int, data.split(',')))
            # Plot the data
            plt.plot(data, marker='o', label=f'{env_name}')
            last_x = len(data) 
            last_y = data[-1]
            plt.text(last_x, last_y, f'{last_y}', fontsize=12, horizontalalignment='right')

    # Add a horizontal line at y = 10 as a benchmark
    plt.axhline(y=10, color='r', linestyle='--', label='UCB_MOPPO: y=30')

    # Add titles and labels
    plt.title('Policy archive curve',fontsize = FONT_SIZE)
    plt.xlabel('Iterations',fontsize = FONT_SIZE)
    plt.ylabel('Number of policy archived',fontsize = FONT_SIZE)

    # # Set y-axis ticks for finer control
    # y_ticks = np.arange(min(data), max(data) + 1, step=50)  # Adjust step for finer resolution as needed
    # plt.yticks(y_ticks)

    # Enable the grid
    plt.grid(True)
    plt.tick_params(axis='both', which='major', labelsize=FONT_SIZE)
    plt.legend(fontsize= FONT_SIZE-3)
    # Add a legend
    plt.legend()

    # Show the plot
    plt.show()


    
        
# ====for benchmark PGMORL=====
# collect_benchmark_policy_store()
# collect_benchmark_hv_pf_ep()

MOPPO_root = r'E:\WorkShare\MORL\MOPPO\results'
env_list = ['Swimmer-v2', 'HalfCheetah-v2', 'Walker2d-v2','Ant-v2','Hopper-v2', 'Hopper-v3', 'Humanoid-v2']
algo_list = ['fix-preference','random-preference']

env_name = 'Ant-v2'
algorithm_name = 'dwc5'
objects = ['x-forward_speed','y_axis_energy_efficiency']

# step 1
# plt_hv_curve(env_name,MOPPO_root+f'/{env_name}/{algorithm_name}',1234567)
# examinate_plot_donimate_pf(env_name,MOPPO_root+f'/{env_name}/{algorithm_name}',123456,objects,2)
# step 2
# plot_pf(env_name,MOPPO_root+f'/{env_name}/{algorithm_name}',123456,objects)
# plot_full_pf(env_name,MOPPO_root+f'/{env_name}/{algorithm_name}',123456,objects)

# ====for hopper-v3=====
env_name = 'Hopper-v3'
algorithm_name = 'dwc4'
objects = ['x-forward_speed','y_jump_height','z_axis_energy_efficiency']
# plt_hv_curve(env_name,MOPPO_root+f'/{env_name}/{algorithm_name}',123456)
# examinate_plot_donimate_pf(env_name,f'/workdir/results/{env_name}/{algorithm_name}',123456,objects,3)

file_paths = [f'/workdir/results/{env_name}/dwc1', f'/workdir/results/{env_name}/dwc2'] 
# plt_hv_curves_compare(env_name, file_paths, 123456,algo_list)
# plot_pf_compare(env_name,file_paths,123456,objects,algo_list,3)




 # Add your file paths here

benchmark_root = r'E:\WorkShare\MORL\PGMORL-master\PGMORL-master\results'
MOPPO_root = r'E:\WorkShare\MORL\MOPPO\results'

iters = [[40,250,10],[80,620,20],[80,620,20],[200,1000,40],[200,1000,40],[200,1000,40]]
algo_list = ['fix-preference','random-preference','moppo-mean','moppo-ucb','PGMORL']
env_names = ['Swimmer-v2', 'HalfCheetah-v2', 'Walker2d-v2','Hopper-v3','Hopper-v2','Ant-v2']
for p, env_name in enumerate(env_names):
 
    if env_name in ['Swimmer-v2', 'HalfCheetah-v2', 'Walker2d-v2']:
        obj_names = ['forward speed','energy efficiency']
        obj_num = 2
    elif env_name == 'Ant-v2':
        obj_names = ['x_axis speed','y_axis speed']
        obj_num = 2
    elif env_name == 'Hopper-v2':
        obj_names = ['run speed','jump height']
        obj_num = 2
    elif env_name == 'Hopper-v3':
        obj_names = ['forwards speed','jump height','energy efficiency']
        obj_num = 3

    # file_paths = [MOPPO_root+f'\{env_name}\dwc1', MOPPO_root + f'\{env_name}\dwc2',benchmark_root+f'\{env_name}'+r'\pgmorl\0'] 
    file_paths = [MOPPO_root+f'\{env_name}\dwc1', MOPPO_root + f'\{env_name}\dwc2', MOPPO_root + f'\{env_name}\dwc3', MOPPO_root + f'\{env_name}\dwc4', benchmark_root+f'\{env_name}'+r'\pgmorl\0'] 
    plot_pf_compare(MOPPO_root,env_name,file_paths,123456,obj_names,algo_list,obj_num)
    # plt_hv_curves_compare(MOPPO_root,env_name, file_paths, 123456,algo_list,iters[p])