from pycuda import driver

def memory_bandwidth(id):
  device = driver.Device(id)
  memory_clock_rate = device.MEMORY_CLOCK_RATE
  bus_width = device.GLOBAL_MEMORY_BUS_WIDTH
  bandwidth = 2.0 * memory_clock_rate * (bus_width/8) / (1 << 20)
  return bandwidth
