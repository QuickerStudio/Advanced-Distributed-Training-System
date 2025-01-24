import psutil
import logging
import time
import pandas as pd
from concurrent.futures import ThreadPoolExecutor
import numpy as np

class AIResourceManager:
    def __init__(self, max_workers=4):
        # Initialize logger
        self.logger = logging.getLogger('AIResourceManager')
        self.logger.setLevel(logging.INFO)
        handler = logging.FileHandler('resource_usage.log')  # Save logs to a file
        handler.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
        
        # Initialize thread pool executor
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self.resource_usage_data = []

        # Set initial goal
        self.initial_goal = "Dynamic resource allocation to efficiently train large AI models"

        # Initialize Q-learning parameters
        self.q_table = {}
        self.alpha = 0.1  # Learning rate
        self.gamma = 0.9  # Discount factor
        self.epsilon = 0.1  # Exploration rate

    def collect_resource_usage(self):
        # Collect current resource usage data
        cpu_usage = psutil.cpu_percent(interval=1)
        memory_usage = psutil.virtual_memory().percent
        disk_usage = psutil.disk_usage('/').percent
        net_info = psutil.net_io_counters()
        bytes_sent = net_info.bytes_sent
        bytes_recv = net_info.bytes_recv

        current_time = time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())

        resource_data = {
            "time": current_time,
            "cpu_usage": cpu_usage,
            "memory_usage": memory_usage,
            "disk_usage": disk_usage,
            "bytes_sent": bytes_sent,
            "bytes_recv": bytes_recv
        }

        self.resource_usage_data.append(resource_data)
        if len(self.resource_usage_data) > 100:  # Maintain a sliding window of the last 100 samples
            self.resource_usage_data.pop(0)

        self.logger.info(f'Time: {current_time}, CPU Usage: {cpu_usage}%, Memory Usage: {memory_usage}%, Disk Usage: {disk_usage}%, Bytes Sent: {bytes_sent}, Bytes Received: {bytes_recv}')

        # Save the data to a log file
        with open('resource_usage.csv', 'a') as f:
            f.write(f"{current_time},{cpu_usage},{memory_usage},{disk_usage},{bytes_sent},{bytes_recv}\n")

        return resource_data

    def choose_action(self, state):
        # Choose action based on epsilon-greedy strategy
        if np.random.uniform(0, 1) < self.epsilon:
            # Explore: choose a random action
            return np.random.choice(["allocate", "deallocate", "no_action"])
        else:
            # Exploit: choose the best action from Q-table
            return max(self.q_table.get(state, {}), key=self.q_table.get(state, {}).get, default="no_action")

    def learn(self, state, action, reward, next_state):
        # Update Q-table based on the action taken and reward received
        q_value = self.q_table.get(state, {}).get(action, 0.0)
        max_next_q_value = max(self.q_table.get(next_state, {}).values(), default=0.0)
        new_q_value = q_value + self.alpha * (reward + self.gamma * max_next_q_value - q_value)
        
        if state not in self.q_table:
            self.q_table[state] = {}
        self.q_table[state][action] = new_q_value

    def allocate_resources(self):
        # Analyze the collected data and predict resource needs
        if len(self.resource_usage_data) < 100:
            return  # Ensure we have enough data

        current_data = self.resource_usage_data[-1]
        state = (current_data['cpu_usage'], current_data['memory_usage'])
        action = self.choose_action(state)

        # Simulate action and receive reward
        if action == "allocate":
            self.logger.info('Allocating more resources...')
            reward = 10  # Example reward for allocating resources
        elif action == "deallocate":
            self.logger.info('Deallocating some resources...')
            reward = -10  # Example reward for deallocating resources
        else:
            self.logger.info('No action taken.')
            reward = 0  # No reward for no action

        # Update Q-table
        next_state = (current_data['cpu_usage'], current_data['memory_usage'])
        self.learn(state, action, reward, next_state)

        # Check if it's idle time to schedule AI training
        idle_times_by_hour = self.find_idle_times()
        current_hour = pd.to_datetime(current_data['time']).hour
        if idle_times_by_hour.get(current_hour, 0) > 0:
            self.logger.info('Idle time detected. Scheduling AI training...')
            # Schedule AI training tasks

    def find_idle_times(self):
        # Read data from the log file
        data = pd.read_csv('resource_usage.csv', names=["time", "cpu_usage", "memory_usage", "disk_usage", "bytes_sent", "bytes_recv"])
        
        # Convert time to datetime format
        data['time'] = pd.to_datetime(data['time'])
        
        # Find idle times based on resource usage
        idle_times = data[(data['cpu_usage'] < 20) & (data['memory_usage'] < 20)]
        
        # Group by hour to find patterns
        idle_times_by_hour = idle_times.groupby(idle_times['time'].dt.hour).size()
        
        self.logger.info(f'Idle times by hour: {idle_times_by_hour}')
        return idle_times_by_hour

    def run(self, interval=10):
        # Periodically collect data and allocate resources
        self.logger.info(f'Initial Goal: {self.initial_goal}')
        while True:
            self.executor.submit(self.collect_resource_usage)
            self.executor.submit(self.allocate_resources)
            time.sleep(interval)

# Example usage
if __name__ == "__main__":
    ai_resource_manager = AIResourceManager()
    ai_resource_manager.run()