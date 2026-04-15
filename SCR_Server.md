# SCR Server Architecture: TORCS UDP Communication Pipeline

## Overview

The **SCR (Simulated Car Racing) Server** is a TORCS driver module that establishes a UDP-based client-server architecture for real-time control and telemetry exchange. This document describes the complete pipeline from TORCS simulation to agent perception and control.

---

## System Architecture

### Process Flow

```
┌─────────────────────────────────────────────────────────────────┐
│ TORCS Race Simulator (Main Process)                             │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │ SCR Server Driver Module (scr_server.cpp)                │   │
│  │ - Runs at each simulation tick (~50 Hz)                  │   │
│  │ - Updates car physics, sensors, track state              │   │
│  └──────────────────────────────────────────────────────────┘   │
│         │ UDP Port 3001+ │                                      │
│         │ (localhost)    │                                      │
│         ▼                ▼                                      │
│      LISTEN            SEND                                     │
│      (Receive          (Transmit                                │
│       commands)        state)                                   │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
         │                          ▲
         │ Init Handshake           │ StateString (UDP)
         │ "SCR(init ...)"          │ "(angle ..)(curLapTime ..).."
         ▼                          │
┌─────────────────────────────────────────────────────────────────┐
│ Client Agent (snakeoil3_gym.py or similar)                      │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │ Connection Handshake                                     │   │
│  │ 1. Send: "SCR(init <track_sensor_angles>)"               │   │
│  │ 2. Recv: "***identified***"                              │   │
│  └──────────────────────────────────────────────────────────┘   │
│         │                                                       │
│  ┌──────▼───────────────────────────────────────────────────┐   │
│  │ Main Control Loop                                        │   │
│  │ for each timestep:                                       │   │
│  │   1. recv(StateString) → parse_server_str()              │   │
│  │   2. Process observation (S.d dictionary)                │   │
│  │   3. Compute agent action                                │   │
│  │   4. send(DriverAction) via UDP                          │   │
│  └──────────────────────────────────────────────────────────┘   │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## Communication Protocol

### Initialization Phase

**1. Client → Server**
```
"SCR(init -45 -19 -12 -7 -4 -2.5 -1.7 -1 -0.5 0 0.5 1 1.7 2.5 4 7 12 19 45)"
```
- Identifies as SCR client
- Provides 19 track sensor angles in degrees
- Sets track sensor configuration before race starts

**2. Server → Client**
```
"***identified***"
```
- Confirms successful identification
- Server is now ready to exchange telemetry

### Steady-State Loop (Every Simulation Tick ~50Hz)

**1. Server → Client: StateString**
```
(angle 0.123)(curLapTime 45.678)(damage 1000)...(focus -1 100 -1 110 -1)
```
- 20 state fields (see details below)
- Optional: vision image data (64×64 pixels encoded)
- Size: ~500B–2KB depending on vision flag

**2. Client → Server: DriverAction**
```
(accel 0.75)(brake 0)(clutch 0)(gear 3)(steer -0.2)(focus 0)(meta 0)
```
- Control commands sent back to TORCS
- Must arrive within timeout (~100ms) or server uses last valid command

---

## StateString Fields: Complete Reference

### Field Data Structure

Each field in the StateString is encoded as:
```
(fieldname value1 value2 ... valueN)
```

Vectors (arrays) are space-separated floats. The server sends fields in a fixed, deterministic order:

| # | Field | Type | Size | Units | Description |
|---|-------|------|------|-------|-------------|
| 1 | `angle` | float | 1 | radians | Car's yaw angle relative to track axis; range [-π, π]. Negative = left turn, positive = right turn. |
| 2 | `curLapTime` | float | 1 | seconds | Elapsed time in current lap. Resets to 0 at lap start. |
| 3 | `damage` | float | 1 | units | Cumulative collision/damage value. If damage limit reached, car is disabled. |
| 4 | `distFromStart` | float | 1 | meters | Distance from track start line for current segment of track. Range: [0, track segment length]. |
| 5 | `totalDistFromStart` | float | 1 | meters | Total cumulative distance traveled including all completed laps. |
| 6 | `distRaced` | float | 1 | meters | Net distance covered in current race episode (computed locally by server). |
| 7 | `fuel` | float | 1 | liters | Remaining fuel in tank. 0 = out of fuel (car stops). |
| 8 | `gear` | int | 1 | enum | Current transmission gear: -1=Reverse, 0=Neutral, 1-6=Forward gears. |
| 9 | `lastLapTime` | float | 1 | seconds | Time taken to complete the last full lap (or 0 if not yet completed). |
| 10 | `opponents` | float[] | 36 | meters | Distance sensors to 36 opponent cars evenly distributed around vehicle (10° increments). -1 if no opponent in direction. |
| 11 | `racePos` | int | 1 | position | Current position in race (1 = leader, N = last place). |
| 12 | `rpm` | float | 1 | rpm × 10 | Engine RPM multiplied by 10. Divide by 10 to get actual RPM. |
| 13 | `speedX` | float | 1 | km/h | Longitudinal speed (forward/backward). Positive = forward, negative = reverse. |
| 14 | `speedY` | float | 1 | km/h | Lateral speed (left/right). Positive = right, negative = left. Indicates slip/drifting. |
| 15 | `speedZ` | float | 1 | km/h | Vertical speed (up/down). Positive = ascending, negative = descending. |
| 16 | `track` | float[] | 19 | meters | Rangefinder distances to track edges/obstacles at 19 fixed angles (from -90° to +90° in 10° steps). -1 = out-of-track or invalid. |
| 17 | `trackPos` | float | 1 | normalized | Position relative to track centerline: -1.0 (left edge) to +1.0 (right edge). 0 = center. -1 if off-track. |
| 18 | `wheelSpinVel` | float[] | 4 | rad/s | Angular velocity of 4 wheels [FL, FR, RL, RR]. High spin velocity ÷ low forward speed = wheel slip/spinout. |
| 19 | `z` | float | 1 | meters | Height of car above track surface. 0 = on track, positive = airborne, negative = underground (error state). |
| 20 | `focus` | float[] | 5 | meters | 5 steerable rangefinder beams centered on `focusCmd` angle (±2° offsets). -1 if in cooldown or out-of-track. Cooldown: ~1 sec after each query. |
| *(optional)* | `img` | byte[] | N | pixel bytes | 64×64 RGB image of driver viewpoint (sent if `vision=True` in gym_torcs). |

### Field Details by Category

#### **Position & Distance**
- **`trackPos`**: Indicates centering. Use to steer toward center: `steer_correction = -trackPos * gain`
- **`distFromStart`**, **`totalDistFromStart`**: For progress-based rewards
- **`distRaced`**: Differential distance for reward calculation

#### **Speed (3-axis)**
- **`speedX`**: Primary control target (throttle/brake adjust to maintain desired speed)
- **`speedY`**: Indicates drifting/lateral slip; high value = unstable
- **`speedZ`**: Airborne detection; non-zero usually indicates crash or jump

#### **Sensors: Track Geometry**
- **`track` (19 beams)**: Always-on fixed-angle rangefinders; shape of track ahead
  - Index 9 = center (0°), indices 0–8 = left side (-90° to -10°), indices 10–18 = right side (10° to 90°)
  - Used by Snakeoil for obstacle avoidance and centering
  - Value `-1` = out-of-bounds; indicates track edge or off-road

- **`focus` (5 beams)**: Selective rangefinders; requires agent to "request" via `focusCmd`
  - 5 beams: center + 2° left + 2° right + 4° left + 4° right
  - Always returned as `-1` during cooldown (~1 sec) after each query
  - **Strategic use**: Agent learns when to "look ahead" at specific angles

#### **Sensors: Opponents**
- **`opponents` (36 beams)**: 360° opponent detection in 10° increments
  - Range: [0, 200m]; -1 if no opponent detected
  - Value `-1` = safe in that direction
  - Combined with agent's lateral position to assess race situation

#### **Engine & Transmission**
- **`gear`**: Discrete enum; server respects `gearCmd` from client but clamps to valid range
- **`rpm`**: Divided by 10 in transmission (multiply by 10 to recover; helps normalize to [0, 10000] range)

#### **Damage & Resources**
- **`damage`**: Accumulates from collisions; if exceeds limit (~10,000), car becomes disabled/jittered
- **`fuel`**: Consumed at ~0.8 L/sec at high speed; race ends if depleted

#### **Timing**
- **`curLapTime`**: Increments each tick; resets at lap boundary
- **`lastLapTime`**: Becomes non-zero only after first lap completion; useful for lap-time-based rewards

---

## Snakeoil Client: Parsing & State Dictionary

### Parse Process

When snakeoil receives the StateString in [snakeoil3_gym.py](snakeoil3_gym.py):

```python
def parse_server_str(self, server_string):
    '''Parse the server string.'''
    self.servstr = server_string.strip()[:-1]
    sslisted = self.servstr.strip().lstrip('(').rstrip(')').split(')(')
    for i in sslisted:
        w = i.split(' ')
        self.d[w[0]] = destringify(w[1:])  # Convert strings to floats/lists
```

**Example Parsing:**

Raw StateString (abbreviated):
```
(angle 0.05)(curLapTime 23.4)(damage 500)...(track 85 90 100 95 ...)(focus -1 -1 50 -1 -1)
```

After parsing, `ServerState.d` dictionary:
```python
S.d = {
    'angle': 0.05,                      # float
    'curLapTime': 23.4,                 # float
    'damage': 500,                      # float
    'track': [85, 90, 100, 95, ...],    # list of 19 floats
    'focus': [-1, -1, 50, -1, -1],      # list of 5 floats
    ...
}
```

### Snakeoil State Dictionary Keys

Access in client code:
```python
S = c.S.d  # State dictionary

# Single-value sensors
angle = S['angle']
speed_x = S['speedX']
track_pos = S['trackPos']
rpm = S['rpm']
gear = S['gear']

# Array sensors
track_distances = S['track']        # [19 floats]
opponent_distances = S['opponents'] # [36 floats]
focus_distances = S['focus']        # [5 floats]
wheel_spins = S['wheelSpinVel']     # [4 floats]

# Timing & race state
lap_time = S['curLapTime']
last_lap = S['lastLapTime']
damage = S['damage']
fuel = S['fuel']
race_pos = S['racePos']
```

---

## Control Pipeline: Agent → TORCS

### DriverAction Format

Snakeoil constructs the response in [snakeoil3_gym.py DriverAction class](snakeoil3_gym.py#L317):

```python
class DriverAction():
    def __init__(self):
        self.d = {
            'accel': 0.2,       # [0, 1] acceleration pedal
            'brake': 0,         # [0, 1] brake pedal
            'clutch': 0,        # [0, 1] clutch engagement
            'gear': 1,          # {-1, 0, 1, 2, 3, 4, 5, 6}
            'steer': 0,         # [-1, 1] steering wheel
            'focus': 0,         # angle or 360 (no request)
            'meta': 0           # {0, 1} restart flag
        }

    def __repr__(self):
        # Encodes as: "(accel 0.2)(brake 0)(clutch 0)(gear 1)(steer 0)(focus 0)(meta 0)"
```

### Server Reception & Execution

In [scr_server.cpp drive() function](vtorcs-RL-color/src/drivers/scr_server/scr_server.cpp#L605):

```cpp
// Parse incoming DriverAction from client
std::string lineStr(line);
CarControl carCtrl(lineStr);

// Apply controls to car physics
car->_accelCmd  = carCtrl.getAccel();   // Throttle input
car->_brakeCmd  = carCtrl.getBrake();   // Brake input
car->_gearCmd   = carCtrl.getGear();    // Transmission
car->_steerCmd  = carCtrl.getSteer();   // Steering
car->_clutchCmd = carCtrl.getClutch();  // Clutch
car->_focusCmd  = carCtrl.getFocus();   // Focus sensor direction

// TORCS physics engine then:
//  1. Updates car position/velocity/acceleration
//  2. Applies aerodynamics, tire friction, collisions
//  3. Updates next tick on StateString with new values
```

---

## Integration with gym_torcs

### Observation Parsing

[gym_torcs.py make_observaton()](gym_torcs.py#L232) converts ServerState to RL observation:

```python
def make_observaton():
    S = c.S.d  # Parsed StateString dictionary
    
    # Build namedtuple observation
    return Observation(
        focus=           S['focus'],           # [5]
        speedX=          S['speedX'],          # scalar
        speedY=          S['speedY'],          # scalar
        speedZ=          S['speedZ'],          # scalar
        opponents=       S['opponents'],       # [36]
        rpm=             S['rpm'],             # scalar
        track=           S['track'],           # [19]
        wheelSpinVel=    S['wheelSpinVel']     # [4]
    )
    # Total: 5 + 1 + 1 + 1 + 36 + 1 + 19 + 4 = 68-element vector when flattened
```

### Action Conversion

[gym_torcs.py step() function](gym_torcs.py#L54) accepts RL action and converts to TORCS command:

```python
def step(u):  # u is RL action (typically steering angle or [steering, throttle])
    # Convert RL action to vehicle command
    # Example: u[0] = steering in [-1, 1]
    action_torcs = { ... }
    
    # Send via snakeoil to SCR server
    c.R.d['steer'] = u[0]
    c.R.d['accel'] = u[1]  # if throttle control enabled
    c.respond_to_server()   # UDP send
```

---

## Key Implementation Files

| File | Purpose | Key Elements |
|------|---------|--------------|
| [scr_server.cpp](vtorcs-RL-color/src/drivers/scr_server/scr_server.cpp) | C++ TORCS driver module | Sensor initialization (lines 287–294), state building (lines 499–521), UDP I/O (line 569) |
| [snakeoil3_gym.py](snakeoil3_gym.py) | Python UDP client | ServerState parser (line 298), DriverAction encoder (line 317), connection loop (line 229) |
| [gym_torcs.py](gym_torcs.py) | Gym environment wrapper | Observation construction (line 232), action mapping (line 54), reward (line 111) |
| [SimpleParser.cpp](vtorcs-RL-color/src/drivers/scr_server/SimpleParser.cpp) | String serialization | `stringify()` method for encoding tuples as `(key val1 val2...)` |

---

## Typical Control Loop Timing

```
t=0.00s    Client sends init message     → Server identifies
     ↓
t=0.02s    TORCS tick 1: Server builds state & sends UDP
     ↓
t=0.021s   Client receives, parses, computes action, responds
     ↓
t=0.04s    TORCS tick 2: Server receives action, updates physics, sends new state
     ↓
t=0.042s   Client receives next state
     ↓
...cycle repeats at ~50 Hz (20ms per tick)
```

**Timeout handling:** If client doesn't respond within the timeout window (~100ms), the last received action is repeated automatically. This prevents the TORCS simulation from stalling.

---

## Sensor Behavior & Gotchas

### Track Sensor Returns -1 When:
1. Car is out-of-bounds (off-track or through barrier)
2. Track edge is too far (beyond rangefinder range ~100–200m)

### Focus Sensor Returns -1 When:
1. Car is out-of-bounds (same as track sensor)
2. Client is in cooldown period (~1 sec) after issuing query
3. Client sends `focusCmd = 360` (explicitly disabled request)

### Noise (Optional):
- Track sensors: ±10% random noise (if noise flag enabled)
- Focus sensors: ±1% random noise
- Opponent sensors: ±2% random noise

