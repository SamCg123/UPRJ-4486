import argparse
import time
import msgpack
from enum import Enum, auto

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

from planning_utils import a_star, heuristic, create_grid, iterative_astar, ucs, a_star2, heuristic2, create_grid_and_edges, closest_point, bfs
from udacidrone import Drone
from udacidrone.connection import MavlinkConnection
from udacidrone.messaging import MsgID
from udacidrone.frame_utils import global_to_local


class States(Enum):
    MANUAL = auto()
    ARMING = auto()
    TAKEOFF = auto()
    WAYPOINT = auto()
    LANDING = auto()
    DISARMING = auto()
    PLANNING = auto()


class MotionPlanning(Drone):

    def __init__(self, connection):
        super().__init__(connection)

        self.target_position = np.array([0.0, 0.0, 0.0])
        self.waypoints = []
        self.in_mission = True
        self.check_state = {}

        # initial state
        self.flight_state = States.MANUAL

        # register all your callbacks here
        self.register_callback(MsgID.LOCAL_POSITION, self.local_position_callback)
        self.register_callback(MsgID.LOCAL_VELOCITY, self.velocity_callback)
        self.register_callback(MsgID.STATE, self.state_callback)

    def local_position_callback(self):
        if self.flight_state == States.TAKEOFF:
            if -1.0 * self.local_position[2] > 0.95 * self.target_position[2]:
                self.waypoint_transition()
        elif self.flight_state == States.WAYPOINT:
            if np.linalg.norm(self.target_position[0:2] - self.local_position[0:2]) < 1.0:
                if len(self.waypoints) > 0:
                    self.waypoint_transition()
                else:
                    if np.linalg.norm(self.local_velocity[0:2]) < 1.0:
                        self.landing_transition()

    def velocity_callback(self):
        if self.flight_state == States.LANDING:
            if self.global_position[2] - self.global_home[2] < 0.1:
                if abs(self.local_position[2]) < 0.01:
                    self.disarming_transition()

    def state_callback(self):
        if self.in_mission:
            if self.flight_state == States.MANUAL:
                self.arming_transition()
            elif self.flight_state == States.ARMING:
                if self.armed:
                    self.plan_path()
            elif self.flight_state == States.PLANNING:
                self.takeoff_transition()
            elif self.flight_state == States.DISARMING:
                if ~self.armed & ~self.guided:
                    self.manual_transition()

    def arming_transition(self):
        self.flight_state = States.ARMING
        print("arming transition")
        self.arm()
        self.take_control()

    def takeoff_transition(self):
        self.flight_state = States.TAKEOFF
        print("takeoff transition")
        self.takeoff(self.target_position[2])

    def waypoint_transition(self):
        self.flight_state = States.WAYPOINT
        print("waypoint transition")
        self.target_position = self.waypoints.pop(0)
        print('target position', self.target_position)
        self.cmd_position(self.target_position[0], self.target_position[1], self.target_position[2], self.target_position[3])

    def landing_transition(self):
        self.flight_state = States.LANDING
        print("landing transition")
        self.land()

    def disarming_transition(self):
        self.flight_state = States.DISARMING
        print("disarm transition")
        self.disarm()
        self.release_control()

    def manual_transition(self):
        self.flight_state = States.MANUAL
        print("manual transition")
        self.stop()
        self.in_mission = False

    def send_waypoints(self):
        print("Sending waypoints to simulator ...")
        data = msgpack.dumps(self.waypoints)
        self.connection._master.write(data)

    def plan_path(self):
        self.flight_state = States.PLANNING
        print("Searching for a path ...")
        TARGET_ALTITUDE = 5
        SAFETY_DISTANCE = 5

        self.target_position[2] = TARGET_ALTITUDE

        # TODO: read lat0, lon0 from colliders into floating point values
        lat0_lon0 = np.loadtxt('colliders.csv', max_rows=1, dtype='str')
        lat0 = np.float64(str(lat0_lon0[1]).replace(",",""))
        lon0 = np.float64(str(lat0_lon0[3]))

        # TODO: set home position to (lon0, lat0, 0)
        self.set_home_position(lon0, lat0, 0)

        print('global home {0}, position {1}, local position {2}'.format(self.global_home, self.global_position, self.local_position))
        # Read in obstacle map
        data = np.loadtxt('colliders.csv', delimiter=',', dtype='float64', skiprows=2)

        # Define a grid for a particular altitude and safety margin around obstacles
        grid, north_offset, east_offset = create_grid(data, TARGET_ALTITUDE, SAFETY_DISTANCE)
        print("North offset = {0}, east offset = {1}".format(north_offset, east_offset))
        # Define starting point on the grid (this is just grid center)
        grid_start = (-north_offset, -east_offset)
           
        # Set goal as some arbitrary position on the grid
        grid_goal = (-north_offset + 10, -east_offset + 10)

        print('Local Start and Goal: ', grid_start, grid_goal)

        #Demo code
        #path, _ = a_star(grid, heuristic, grid_start, grid_goal)

        #Please uncomment them one by one to validate the answer
        #============================Q1============================
        path, _ = iterative_astar(grid, heuristic, grid_start, grid_goal)

        #============================Q2============================
        #path, _ = ucs(grid, heuristic, grid_start, grid_goal)

        #============================Q3============================
        """ 
        point1 = (-north_offset + 10, -east_offset + 0)
        point2 = (-north_offset + 5, -east_offset + 5)
        point3 = (-north_offset + 0, -east_offset + 10)
        print('Point 1: ', point1)
        print('Point 2: ', point2)
        print('Point 3: ', point3)
        path, _ = a_star2(grid, heuristic2, grid_start, grid_goal, point1, point2, point3) 
        """

        #============================Q4============================
        '''
        #Set a farther goal for clear path on graph
        grid_goal = (-north_offset + 100, -east_offset + 100)

        # Define a flying altitude (feel free to change this) 
        drone_altitude = 5 
        safety_distance = 5 
        grid, edges = create_grid_and_edges(data, drone_altitude, safety_distance) 
        print('Found %5d edges' % len(edges)) 
        
        #Create graph
        G = nx.Graph()
        for e in edges:
            p1 = e[0]
            p2 = e[1]
            dist = np.linalg.norm(np.array(p2) - np.array(p1))
            G.add_edge(p1, p2, weight=dist)
        
        #p1 = (9, 1.2) 
        #p2 = (20, 30) 
        #G = nx.add_edge(p1, p2)
         
        #p1 = (10, 2.2) 
        #p2 = (50, 40) 
        #dist = LA.norm(np.array(p2) - np.array(p1)) 
        #G = nx.add_edge(p1, p2, weight=dist) 
        
        # Find closest point to the graph for start
        graph_start = closest_point(G, grid_start)

        # Find closest point to the graph for goal
        graph_goal = closest_point(G, grid_goal)
        print('Graph Start and Goal: ', graph_start, graph_goal)

        path, _ = bfs(G, heuristic, graph_start, graph_goal)  
        
        # equivalent to 
        # plt.imshow(np.flip(grid, 0)) 
        # Plot it up! 
        plt.imshow(grid, origin='lower', cmap='Greys') 

        for e in edges: 
            p1 = e[0] 
            p2 = e[1] 
            plt.plot([p1[1], p2[1]], [p1[0], p2[0]], 'b-') 
            
        plt.plot([graph_start[1], graph_start[1]], [graph_start[0], graph_start[0]], 'r-')
        for i in range(len(path)-1):
            p1 = path[i]
            p2 = path[i+1]
            plt.plot([p1[1], p2[1]], [p1[0], p2[0]], 'r-')
        plt.plot([graph_goal[1], graph_goal[1]], [graph_goal[0], graph_goal[0]], 'r-')
            
        plt.plot(graph_start[1], graph_start[0], 'rx')
        plt.plot(graph_goal[1], graph_goal[0], 'rx')

        plt.xlabel('EAST')
        plt.ylabel('NORTH')
        plt.show() #Program stop after calling .show()
        '''
        #============================End of Q4============================
        

        # Convert path to waypoints
        waypoints = [[p[0] + north_offset, p[1] + east_offset, TARGET_ALTITUDE, 0] for p in path]
        # Set self.waypoints
        self.waypoints = waypoints
        # TODO: send waypoints to sim (this is just for visualization of waypoints)
        self.send_waypoints()

    def start(self):
        self.start_log("Logs", "NavLog.txt")

        print("starting connection")
        self.connection.start()

        # Only required if they do threaded
        # while self.in_mission:
        #    pass

        self.stop_log()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, default=5760, help='Port number')
    parser.add_argument('--host', type=str, default='127.0.0.1', help="host address, i.e. '127.0.0.1'")
    args = parser.parse_args()

    conn = MavlinkConnection('tcp:{0}:{1}'.format(args.host, args.port), timeout=60)
    drone = MotionPlanning(conn)
    time.sleep(1)

    drone.start()
