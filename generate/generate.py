from weather import generate_weather
from holiday import generate_holiday
from hop import generate_hop
from period import generate_period
from weekend import generate_weekend

DATASET = 'PEMS04'
# hop
HOP_K = 1
# period
TOP_K = 3
NUM_PERIOD = 10

print("generate weekend data...")
generate_weekend(DATASET)

print("generate weather data...")
generate_weather(DATASET)

print("generate hop data...")
generate_hop(DATASET,HOP_K)

print("generate period data...")
generate_period(DATASET,TOP_K,NUM_PERIOD)

print("generate holiday data...")
generate_holiday(DATASET)

