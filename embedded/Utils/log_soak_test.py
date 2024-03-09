#### soak test libraries
import logging
import psutil
import shutil
import time

LOG_LEVEL = logging.DEBUG  # Change to logging.DEBUG to enable print logs

# Configuration of logging variables for the soak-test
class soak_test:
    def __init__(self, log_name="model_soak_test") -> None:
        self.set_name(log_name)
        self.preprocessing_latency = 0
        self.predict_latency = 0
        self.model_accuracy = 0

    def set_name(self, log_name):
        # Configuring logging settings
        logging.basicConfig(filename=log_name+'.log', level=LOG_LEVEL, format='%(asctime)s - %(levelname)s - %(message)s - %(levelname)s - %(asctime)s - %(asctime)s - %(levelname)s')

    def get_cpu_temperature_linux(self):
        # Read the file containing CPU temperature information
        with open("/sys/class/thermal/thermal_zone0/temp", "r") as f:
            temp_str = f.read()
            temp_celsius = int(temp_str) / 1000.0  # Convert to Celsius
            return temp_celsius

    def get_system_variables(self):
        try:
            cpu_temp = self.get_cpu_temperature_linux()
        except:
            cpu_temp = 0

        cpu_usage = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        memory_usage = memory.percent
        disk_usage = shutil.disk_usage("/")
        # total_disk_space = disk_usage.total
        used_disk_space = disk_usage.used
        return cpu_temp, cpu_usage, memory_usage, used_disk_space

    def set_model_performance(self, preprocessing_latency, predict_latency, model_accuracy):
        self.preprocessing_latency = preprocessing_latency
        self.predict_latency = predict_latency
        self.model_accuracy = model_accuracy

    # Function to register system's information and conditions
    def log_info(self):
        cpu_temp, cpu_usage, memory_usage, used_disk_space = self.get_system_variables()
        log_message = f"CPU Usage: {cpu_usage}% | CPU Temperature: {cpu_temp}°C% | Memory Usage: {memory_usage}% | Used Disk Space: {used_disk_space/1000000000} Gbytes | Preprocessing Latency: {self.preprocessing_latency*1000}ms | Model Predict Latency: {self.predict_latency*1000}ms | Model accuracy: {self.model_accuracy} %"

        print(log_message)
        logging.debug(log_message)
        # logging.info(log_message)
