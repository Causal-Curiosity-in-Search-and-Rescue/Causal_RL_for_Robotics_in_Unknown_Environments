import json
import os 
import numpy as np
import pickle
import copy
import random
import matplotlib.pyplot as plt

def generate_maze_with_objects(height, width, num_00, num_01,num_10,num_11,num_s,CONFIG):
    maze1 = generate_maze(height,width,CONFIG)
    maze = copy.deepcopy(maze1)

    # Generate random positions for 'm'
    # for _ in range(num_m):
    #     while True:
    #         row = random.randint(1, height - 2)
    #         col = random.randint(1, width - 2)
    #         if maze[row][col] == 'c':
    #             maze[row][col] = 'm'
    #             break
 
    # # Generate random positions for 'i'
    # for _ in range(num_i):
    #     while True:
    #         row = random.randint(1, height - 2)
    #         col = random.randint(1, width - 2)
    #         if maze[row][col] == 'c':
    #             maze[row][col] = 'i'
    #             break

    # Generate random positions for 's' Start position
    for _ in range(num_s):
        while True:
            row = random.randint(1, height - 2)
            col = random.randint(1, width - 2)
            if maze[row][col] == 'c':
                maze[row][col] = 's'
                break

    ##  Generating Positions for Third variable
    # Generate random positions for '00' [Texture 0, Shape 0]
    for _ in range(num_00):
        while True:
            row = random.randint(1, height - 2)
            col = random.randint(1, width - 2)
            if maze[row][col] == 'c':
                maze[row][col] = '00'
                break
    
    # Generate random positions for '01' [Texture 0, Shape 1]
    for _ in range(num_01):
        while True:
            row = random.randint(1, height - 2)
            col = random.randint(1, width - 2)
            if maze[row][col] == 'c':
                maze[row][col] = '01'
                break
    # Generate random positions for '10' [Texture 1, Shape 0]
    for _ in range(num_10):
        while True:
            row = random.randint(1, height - 2)
            col = random.randint(1, width - 2)
            if maze[row][col] == 'c':
                maze[row][col] = '10'
                break
    
    # Generate random positions for '11' [Texture 1, Shape 1]
    for _ in range(num_11):
        while True:
            row = random.randint(1, height - 2)
            col = random.randint(1, width - 2)
            if maze[row][col] == 'c':
                maze[row][col] = '11'
                break
    
    mkb = CONFIG['environment']['env_codes']['mkb']
    ukb = CONFIG['environment']['env_codes']['ukb']
    
    m_kb_array = convert_to_movable_knowledge_array(maze,mkb)
    u_kb_array = convert_to_no_movable_knowledge_array(maze,ukb)
    
    return maze,m_kb_array,u_kb_array

def convert_to_movable_knowledge_array(lst,mapping):
    
    # Define a mapping for replacement
    #OLD METHOD
    #mapping = {'w': 0, 'u': 1, 'c': 1, 'm': 2, 'i': 3, 'r': 0, 's': 5, 'o': 6}

    #NEW METHOD
    #mapping = {"w":0,"u":1,"c":1,"00":2,"01":3,"10":4,"11":5,"r":0, "s":6,"o":7}
    
    # Replace each element in the list according to the mapping
    replaced_list = [[mapping[item] for item in sublist] for sublist in lst]
    
    # Convert the list to a numpy array
    return np.array(replaced_list,dtype=np.int32)

def convert_to_no_movable_knowledge_array(lst,mapping):


    # Define a mapping for replacement
    #mapping = {'w': 0, 'u': 1, 'c': 1, 'm': 2, 'i': 2, 'r': 0, 's': 5, 'o': 6}
    
    # Replace each element in the list according to the mapping
    replaced_list = [[mapping[item] for item in sublist] for sublist in lst]
    
    # Convert the list to a numpy array
    return np.array(replaced_list,dtype=np.int32)

def visualisemaze(maze,log_dir):
    # Convert maze to numpy array for visualization
    maze_np = np.array([[1 if cell == 'c' else 2 if cell == 'm' else 3 if cell == 'i' else 4 if cell == 'o' else 5 if cell == 's' else 0 for cell in row] for row in maze])

    # Define custom color map
    cmap = plt.cm.viridis
    cmap.set_over('orange')

    plt.imshow(maze_np, cmap=cmap, interpolation='nearest')

    # Customizing color bar for legend
    plt.colorbar(ticks=[0, 1, 2, 3, 4, 5], label='Legend', values=[0, 1, 2, 3, 4, 5], format=plt.FuncFormatter(lambda val, loc: ['walls', 'free space', 'moveable objects', 'immovable objects', 'goal', 'starting position'][int(val)]))

    plt.axis('off')
    # plt.show(block=False)
    plt.savefig(os.path.join(log_dir,'2d_test_env.png'))
    
def generate_maze(height, width,CONFIG):
    maze = [['w' if i == 0 or i == height - 1 or j == 0 or j == width - 1 else 'c' for j in range(width)] for i in range(height)]
    if CONFIG["environment"]["random_room_pos"]:
        maze_with_array = place_array_random(maze,CONFIG)
    else:
        maze_with_array = place_array_in_middle(maze,CONFIG)
    maze_with_u = place_u_near_walls(maze_with_array)
    return maze_with_u

def load_saved_map(map_path):
    with open(map_path,"rb") as file:
        map_plan = pickle.load(file)
    
    return map_plan,convert_to_movable_knowledge_array(map_plan),convert_to_no_movable_knowledge_array(map_plan)
 

def place_array_random(maze,CONFIG):

    array_to_place = CONFIG["environment"]["env_codes"]["room_array"]
    room_arr_np = np.array(array_to_place)
    room_shape = room_arr_np.shape[0]
    start_row = np.random.randint(1,CONFIG['environment']['grid_size'] - room_shape)
    start_col = np.random.randint(1,CONFIG['environment']['grid_size'] - room_shape)

    for i in range(room_shape):
        for j in range(room_shape):
            maze[start_row + i][start_col + j] = array_to_place[i][j]

    return maze

def place_array_in_middle(maze,CONFIG):

    middle_row = len(maze) // 2
    middle_col = len(maze[0]) // 2
 
    array_to_place = CONFIG["environment"]["env_codes"]["room_array"]
 
    for i in range(len(array_to_place)):
        for j in range(len(array_to_place[0])):
            maze[middle_row - 2 + i][middle_col - 3 + j] = array_to_place[i][j]
 
    return maze

def place_u_near_walls(maze):
    height = len(maze)
    width = len(maze[0])
 
    for i in range(1, height - 1):
        for j in range(1, width - 1):
            if maze[i][j] == 'c':
                adjacent_cells = [(i-1, j), (i+1, j), (i, j-1), (i, j+1)]
                for x, y in adjacent_cells:
                    if maze[x][y] == 'w':
                        maze[i][j] = 'u'
 
    return maze

def read_config(config_path='config.json'):
    with open(config_path,"r") as cf:
        data = json.load(cf)
    return data

def modify_config(mod_config,config_path='config.json'):
    with open(config_path,"w") as cf:
        json.dump(mod_config,cf,indent=4)
    config = read_config(config_path=config_path)
    return config

def check_and_create_directory(path):
    """Check if a directory exists at the specified path, and create it if it doesn't exist."""
    if not os.path.exists(path):
        os.makedirs(path)
        print(f"Directory created at: {path}")
    else:
        print(f"Directory already exists at: {path}")
    return path


