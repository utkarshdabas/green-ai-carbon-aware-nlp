from codecarbon import EmissionsTracker
import time

tracker = EmissionsTracker(
    project_name="GreenAI_Mac_Intel",
    measure_power_secs=1
)

tracker.start()

# Simulate CPU work
for _ in range(30_000_000):
    pass

emissions = tracker.stop()

print("Estimated CO2 emissions (kg):", emissions)


    



