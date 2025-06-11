import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Button, RadioButtons
import heapq

# Shape configurations
def get_dog_shape():
    return {
        'positions': {
            1: [0, 2], 2: [0, 1], 3: [0, 0], 4: [1, 1], 5: [2, 2], 6: [2, 1], 7: [2, 0],
            8: [-0.5, 2], 9: [-0.5, 0], 10: [2.5, 2], 11: [2.5, 0],
        },
        'connections': [(1, 2), (2, 3), (2, 4), (4, 6), (5, 6), (6, 7),
                        (1, 8), (3, 9), (5, 10), (7, 11)],
        'center': 4,
        'type': 'dog'
    }

def get_snake_shape():
    return {
        'positions': {i + 1: [i, 5 + np.sin(i * 0.5)] for i in range(12)},
        'connections': [(i, i + 1) for i in range(1, 12)],
        'center': 12,
        'type': 'snake'
    }

def get_line_shape():
    return {
        'positions': {i + 1: [i, 5] for i in range(12)},
        'connections': [(i, i + 1) for i in range(1, 12)],
        'center': 12,
        'type': 'line'
    }

# Globals
obstacle_types = {'High Wall': 'red', 'Tunnel Wall': 'blue', 'Low Wall': 'yellow'}
current_obstacle_type = 'High Wall'
current_shape_type = 'dog'
current_shape = get_dog_shape()

initial_positions = current_shape['positions']
connections = current_shape['connections']
center_id = current_shape['center']
positions = {k: np.array(v, dtype=float) for k, v in initial_positions.items()}

grid_size = (40, 40)
obstacles = {}
path = []
step_index = 0
safety_margin = 1
expanded_obstacles = set()

# Matplotlib setup
fig, ax = plt.subplots()
ax.set_aspect('equal')
ax.set_xlim(0, 40)
ax.set_ylim(0, 40)
ax.grid(True)
dots, = ax.plot([], [], 'o', color='blue')
lines = [ax.plot([], [], 'k-', linewidth=2)[0] for _ in connections]
obstacle_patches = []
goal_marker, = ax.plot([], [], 'gx', markersize=12, label='Goal')

# UI Buttons
axdog = plt.axes([0.75, 0.9, 0.1, 0.05])
dog_btn = Button(axdog, 'Dog')
dog_btn.on_clicked(lambda e: switch_shape(get_dog_shape()))

axsnake = plt.axes([0.75, 0.83, 0.1, 0.05])
snake_btn = Button(axsnake, 'Snake')
snake_btn.on_clicked(lambda e: switch_shape(get_snake_shape()))

axline = plt.axes([0.75, 0.76, 0.1, 0.05])
line_btn = Button(axline, 'Line')
line_btn.on_clicked(lambda e: switch_shape(get_line_shape()))

# Radio Buttons for obstacle type
def set_obstacle_type(label):
    global current_obstacle_type
    current_obstacle_type = label

axradio = plt.axes([0.75, 0.6, 0.15, 0.2])
radio_btn = RadioButtons(axradio, list(obstacle_types.keys()))
radio_btn.on_clicked(set_obstacle_type)

# Heuristic
def heuristic(a, b):
    return np.hypot(a[0] - b[0], a[1] - b[1])

# A* with full-body check
def a_star(start, goal, obstacles, grid_size):
    directions = [(-1,0), (1,0), (0,-1), (0,1), (-1,-1), (-1,1), (1,-1), (1,1)]
    frontier = [(0, start)]
    came_from = {start: None}
    cost_so_far = {start: 0}

    def is_blocked(cell):
        otype = obstacles.get(cell)
        if current_shape_type == 'line':
            return otype is not None
        if otype == 'High Wall':
            return True
        if current_shape_type == 'dog' and otype == 'Tunnel Wall':
            return True
        if current_shape_type == 'snake' and otype == 'Low Wall':
            return True
        return False

    expanded_obstacles.clear()
    for ox, oy in obstacles:
        if is_blocked((ox, oy)):
            for dx in range(-safety_margin, safety_margin + 1):
                for dy in range(-safety_margin, safety_margin + 1):
                    nx, ny = ox + dx, oy + dy
                    if 0 <= nx < grid_size[0] and 0 <= ny < grid_size[1]:
                        expanded_obstacles.add((nx, ny))

    relative_offsets = {
        k: np.array(initial_positions[k]) - np.array(initial_positions[center_id])
        for k in initial_positions
    }

    def is_valid_position(center):
        for offset in relative_offsets.values():
            dot_pos = center + offset
            cx, cy = int(round(dot_pos[0])), int(round(dot_pos[1]))
            if not (0 <= cx < grid_size[0] and 0 <= cy < grid_size[1]):
                return False
            if (cx, cy) in expanded_obstacles:
                return False
        return True

    while frontier:
        _, current = heapq.heappop(frontier)
        if current == goal:
            break
        for dx, dy in directions:
            neighbor = (current[0] + dx, current[1] + dy)
            if not (0 <= neighbor[0] < grid_size[0] and 0 <= neighbor[1] < grid_size[1]):
                continue
            if not is_valid_position(np.array(neighbor, dtype=float)):
                continue
            new_cost = cost_so_far[current] + np.hypot(dx, dy)
            if neighbor not in cost_so_far or new_cost < cost_so_far[neighbor]:
                cost_so_far[neighbor] = new_cost
                priority = new_cost + heuristic(goal, neighbor)
                heapq.heappush(frontier, (priority, neighbor))
                came_from[neighbor] = current

    path = []
    curr = goal
    while curr and curr in came_from:
        path.append(np.array(curr, dtype=float))
        curr = came_from[curr]
    path.reverse()
    return path

# Robot updates
def update_positions():
    global positions, path, step_index
    if not path or step_index >= len(path): return
    next_pos = path[step_index]
    step_index += 1
    delta = next_pos - positions[center_id]
    for k in positions:
        offset = np.array(initial_positions[k]) - np.array(initial_positions[center_id])
        positions[k] = next_pos + offset

    cx, cy = map(int, positions[center_id])
    for dx in range(-safety_margin, safety_margin + 1):
        for dy in range(-safety_margin, safety_margin + 1):
            nx, ny = cx + dx, cy + dy
            if (nx, ny) in obstacles:
                otype = obstacles[(nx, ny)]
                if otype == 'Tunnel Wall' and current_shape_type != 'snake':
                    print('Shroud: switching to snake')
                    switch_shape(get_snake_shape())
                    return
                elif otype == 'Low Wall' and current_shape_type != 'dog':
                    print('Shroud: switching to dog')
                    switch_shape(get_dog_shape())
                    return

# Visualization
def draw_obstacles():
    for patch in obstacle_patches:
        patch.remove()
    obstacle_patches.clear()
    for (ox, oy), otype in obstacles.items():
        rect = plt.Rectangle((ox, oy), 1, 1, color=obstacle_types[otype])
        ax.add_patch(rect)
        obstacle_patches.append(rect)
    for ox, oy in expanded_obstacles:
        if (ox, oy) not in obstacles:
            nearest_type = None
            for dx in range(-safety_margin, safety_margin + 1):
                for dy in range(-safety_margin, safety_margin + 1):
                    neighbor = (ox + dx, oy + dy)
                    if neighbor in obstacles:
                        nearest_type = obstacles[neighbor]
                        break
                if nearest_type:
                    break
            margin_color = obstacle_types.get(nearest_type, 'pink')
            margin_rect = plt.Rectangle((ox, oy), 1, 1, color=margin_color, alpha=0.3)
            ax.add_patch(margin_rect)
            obstacle_patches.append(margin_rect)
            if nearest_type in ['Tunnel Wall', 'Low Wall']:
                edge_color = 'cyan' if nearest_type == 'Tunnel Wall' else 'orange'
                outline = plt.Rectangle((ox, oy), 1, 1, edgecolor=edge_color, facecolor='none',
                                        linewidth=1.5, linestyle='--', alpha=0.6)
                ax.add_patch(outline)
                obstacle_patches.append(outline)

def on_click(event):
    global path, step_index
    if event.inaxes != ax: return
    gx, gy = int(event.xdata), int(event.ydata)
    cell = (gx, gy)
    if event.button == 3:
        start = tuple(map(int, positions[center_id]))
        path.clear()
        new_path = a_star(start, cell, obstacles, grid_size)
        if new_path:
            path.extend(new_path)
            goal_marker.set_data([cell[0] + 0.5], [cell[1] + 0.5])
            step_index = 0
        else:
            print(f"No valid path to {cell}")
    elif event.button == 2 or (event.button == 1 and event.key == 'shift'):
        if cell in obstacles: del obstacles[cell]
        else: obstacles[cell] = current_obstacle_type
        draw_obstacles()

def switch_shape(new_shape):
    global initial_positions, positions, connections, center_id, lines, current_shape_type
    initial_positions = new_shape['positions']
    connections[:] = new_shape['connections']
    center_id = new_shape['center']
    current_shape_type = new_shape['type']
    positions.clear()
    positions.update({k: np.array(v, dtype=float) for k, v in initial_positions.items()})
    for line in lines: line.remove()
    lines[:] = [ax.plot([], [], 'k-', linewidth=2)[0] for _ in connections]

def init(): return dots, *lines

def animate(frame):
    update_positions()
    x_vals = [positions[k][0] for k in positions]
    y_vals = [positions[k][1] for k in positions]
    dots.set_data(x_vals, y_vals)
    for i, (a, b) in enumerate(connections):
        lines[i].set_data([positions[a][0], positions[b][0]], [positions[a][1], positions[b][1]])
    draw_obstacles()
    return dots, *lines, *obstacle_patches, goal_marker

fig.canvas.mpl_connect('button_press_event', on_click)
ani = FuncAnimation(fig, animate, init_func=init, frames=1000, interval=50, blit=True)
plt.title("Shape Navigation with A* and Full-Body Obstacle Avoidance")
handles, labels = ax.get_legend_handles_labels()
if handles:
    plt.legend(handles, labels, loc='upper right')
plt.show()
