import sys
sys.path.append('/home/rhos/umipolicy/lerobot/src')
from lerobot.teleoperators.koch_leader.config_koch_leader import KochLeaderConfig
from lerobot.teleoperators.koch_leader.koch_leader import KochLeader

cfg = KochLeaderConfig(port='/dev/ttyUSB0')
leader = KochLeader(cfg)
leader.connect(calibrate=False)
raw_ticks = leader.bus.sync_read('Present_Position', normalize=False)
homing_offsets = leader.bus.sync_read('Homing_Offset', normalize=False)

print("Raw Ticks:", raw_ticks)
print("Homing Offsets:", homing_offsets)

actual_positions = {k: raw_ticks[k] - homing_offsets[k] for k in raw_ticks}
print("Actual Positions (Mechanical):", actual_positions)
leader.disconnect()
