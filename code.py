import numpy as np
import matplotlib.pyplot as plt

plt.ion()

# Link lengths
link1 = 1.0
link2 = 1.0
link3 = 0.5

# Control parameters
kp = 25
dt = 0.01

# Initial joint angles
theta1 = np.pi / 2
theta2 = np.pi / 2
theta3 = -np.pi / 2

# Goal position
goal_x = 1.0
goal_y = 1.0

# Running flag and path
is_running = True
path_history = [[], []]

# USER INTERACTION


def handle_click(event):
    global goal_x, goal_y
    goal_x, goal_y = event.xdata, event.ydata
    move_to_goal(goal_x, goal_y)

def handle_key(event):
    global is_running, path_history
    if event.key.lower() == 'e':
        is_running = False
        plt.close()
    elif event.key.lower() == 'c':
        path_history = [[], []]

# INVERSE KINEMATICS (3 DOF)


def inverse_kinematics(x, y):
    try:
        e = 1e-5  # small number to avoid division errors

      
        total_length = np.sqrt(x**2 + y**2)
        if total_length > (link1 + link2 + link3):
            print("Target unreachable!")
            return False, None, None, None

        
        x_adj = x - link3 * (x / total_length)
        y_adj = y - link3 * (y / total_length)

      
        cos_theta2 = (x_adj**2 + y_adj**2 - link1**2 - link2**2) / (2 * link1 * link2)
        theta2 = np.arccos(np.clip(cos_theta2, -1, 1))

        r = np.sqrt(x_adj**2 + y_adj**2)
        phi = np.arctan2(y_adj, x_adj + e)
        beta = np.arcsin((link2 * np.sin(theta2)) / (r + e))
        theta1 = phi - beta

      
        theta3 = -(theta1 + theta2)

        return True, theta1, theta2, theta3

    except Exception as e:
        print("IK failed:", e)
        return False, None, None, None


# MOTION CONTROL


def move_to_goal(x, y):
    global theta1, theta2, theta3, is_running

    success, goal_theta1, goal_theta2, goal_theta3 = inverse_kinematics(x, y)
    if not success:
        return

    while is_running:
        # Proportional control for joint movement
        theta1 += kp * (goal_theta1 - theta1) * dt
        theta2 += kp * (goal_theta2 - theta2) * dt
        theta3 += kp * (goal_theta3 - theta3) * dt

        draw_arm(theta1, theta2, theta3, x, y)


# DRAW ARM


def draw_arm(theta1, theta2, theta3, x_goal, y_goal):
    global path_history

    # Forward kinematics
    shoulder = np.array([0, 0])
    joint1 = shoulder + link1 * np.array([np.cos(theta1), np.sin(theta1)])
    joint2 = joint1 + link2 * np.array([np.cos(theta1 + theta2), np.sin(theta1 + theta2)])
    end_effector = joint2 + link3 * np.array([np.cos(theta1 + theta2 + theta3),
                                              np.sin(theta1 + theta2 + theta3)])

    path_history[0].append(end_effector[0])
    path_history[1].append(end_effector[1])

    # Plotting
    plt.cla()

  
    domain_x = np.linspace(-link1 - link2 - link3, link1 + link2 + link3, 200)
    domain_y = np.sqrt(np.clip((link1 + link2 + link3)**2 - domain_x**2, 0, None))
    plt.plot(domain_x, domain_y, 'r--')

    
    xs = [shoulder[0], joint1[0], joint2[0], end_effector[0]]
    ys = [shoulder[1], joint1[1], joint2[1], end_effector[1]]
    plt.plot(xs, ys, 'k-o')

    plt.plot(path_history[0], path_history[1], 'b--')

    plt.title("3-DOF Arm | Click to Move | 'c' to Clear | 'e' to Exit")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.xlim(-link1 - link2 - link3, link1 + link2 + link3)
    plt.ylim(-0.5, link1 + link2 + link3)
    plt.grid(True)
    plt.pause(dt)

# MAIN FUNCTION


def main():
    fig = plt.figure()
    fig.canvas.mpl_connect('button_press_event', handle_click)
    fig.canvas.mpl_connect('key_press_event', handle_key)
    move_to_goal(goal_x, goal_y)

if __name__ == '__main__':
    main()
