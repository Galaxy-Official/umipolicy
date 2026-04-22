# Flexiv RDK 1.0+ Migration

I have updated the codebase to resolve the Python 3.10 / 3.12 compatibility issues. The old `sys.path` injection that forced the codebase to load the legacy Python 3.10 bindings from `lib_py` has been removed. The codebase now natively interfaces with the new RDK 1.0+ Python bindings!

## What was changed:
1. **Removed `sys.path.append`**: Scripts like `read_current_pos.py`, `lerobot_flexiv.py`, and `flexiv_simple_env.py` no longer attempt to force-load the old `flexivrdk` library.
2. **Updated `FlexivInterface` API**: 
   - Uses `robot.states().q` instead of `robot.getRobotStates()`.
   - Uses `robot.SwitchMode()` instead of `robot.setMode()`.
   - Uses `spdlog.ConsoleLogger` instead of `flexivrdk.Log()`.
   - Updated syntax for `ClearFault()`, `Enable()`, `operational()`, `busy()`, `ExecutePrimitive()`, `SendJointPosition()`, and `Move()`.
3. **Robot Connection Protocol**: The new RDK strictly requires a **Serial Number** (e.g. `Rizon4-123456`) rather than an IP address.

## Action Required Before Running:
1. Ensure the new Python 3.12 `flexivrdk` wheel is installed in your environment.
2. Install `spdlog` if you haven't already:
   ```bash
   pip install spdlog
   ```
3. Set your robot's Serial Number in your environment before running the scripts:
   ```bash
   export FLEXIV_ROBOT_SN="Rizon4s-123456" # Replace with your robot's actual SN
   ```
