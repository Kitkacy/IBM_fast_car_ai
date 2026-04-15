# SCR Server Communication Notes (TORCS + Python Client)

This document describes what the current SCR server implementation actually does in this repository, with emphasis on:
- server initialization
- UDP connection/identification handshake
- state/action parsing and serialization

Code references:
- `vtorcs-RL-color/src/drivers/scr_server/scr_server.cpp`
- `snakeoil3_gym.py`

## 1. Server Initialization Path

The SCR module is exposed by `scr_server(...)`, then wired through `InitFuncPt(...)` to TORCS callbacks:
- `rbNewTrack -> initTrack`
- `rbNewRace -> newrace`
- `rbDrive -> drive`
- `rbEndRace -> endrace`
- `rbShutdown -> shutdown`

Important initialization details in `newrace(...)`:
1. Timeout source:
   - `UDP_TIMEOUT` default is `100000` microseconds (`100 ms`).
   - It can be overridden by `getTimeout()` if positive.
2. Sensor range selection by SCR version:
   - `2009 -> 100`
   - `2010/2011 -> 200`
3. UDP socket setup:
   - server creates a UDP socket (`socket(AF_INET, SOCK_DGRAM, 0)`)
   - binds to `getUDPListenPort() + index`
4. Identification loop:
   - blocks in `recvfrom(...)` until a valid init packet is accepted
5. After identification:
   - allocates focus/track/opponent sensors
   - initializes track sensors with angles from client init packet (or fallback defaults)

## 2. UDP Identification Handshake

### What the server checks

In `newrace(...)`, identification condition is:
- `strncmp(line, UDP_ID, 3) == 0`

`UDP_ID` is defined as `"SCR-VIS"`, but only the first 3 characters are checked.
So anything starting with `"SCR"` passes this check (for example `SCR(init ...)`).

### Init payload parsing

After ID check passes, server parses:
- `SimpleParser::parse(initStr, "init", trackSensAngle[index], 19)`

If parsing fails, it falls back to fixed angles:
- `-90, -80, ..., 80, 90`

If parsing succeeds, those 19 client-provided angles are used for track sensors.

### Server acknowledgment

When identification succeeds, server sends:
- `***identified***`

The Python client (`Client.setup_connection`) loops sending init packets until it receives this token.

## 3. Per-Tick Runtime Flow (drive)

In each `drive(...)` call, the server does this in order:
1. update sensor values and derived quantities
2. build full state string
3. send state string to client via `sendto(...)`
4. wait up to `UDP_TIMEOUT` for client action using `select(...)`
5. if action is received:
   - parse with `CarControl(lineStr)`
   - apply controls to car (`accel`, `brake`, `gear`, `steer`, `clutch`, `focus`)
6. if timeout/no action:
   - reuse previously applied controls (`oldAccel`, `oldBrake`, `oldGear`, `oldSteer`, `oldClutch`, `oldFocus`)

So the server sends state first, then reads action for the next control application.

## 4. State String Contents and Order

The state string is serialized in this exact order:
1. `angle`
2. `curLapTime`
3. `damage`
4. `distFromStart`
5. `totalDistFromStart`
6. `distRaced`
7. `fuel`
8. `gear`
9. `lastLapTime`
10. `opponents` (36)
11. `racePos`
12. `rpm` (sent as `car->_enginerpm * 10`)
13. `speedX` (km/h)
14. `speedY` (km/h)
15. `speedZ` (km/h)
16. `track` (19)
17. `trackPos`
18. `wheelSpinVel` (4)
19. `z`
20. `focus` (5)
21. optional `img` (only when vision mode is enabled)

Notes:
- `track` sensors are based on init angles (not always fixed -90..90).
- `focus` values are `-1` while in cooldown or when `focusCmd == 360`.
- If the car is outside track bounds (`trackPos` beyond [-1, 1]), both `track` and `focus` are forced to `-1`.

## 5. Distances: What Is Actually Computed

The server computes:
- `distFromStart`: `car->race.distFromStartLine`
- `totalDistFromStart`: `track_length * (laps - 1) + distFromStartLine`
- `distRaced`: incrementally accumulated local value with wrap handling around lap transitions

`distRaced` is not directly TORCS-provided; it is server-maintained (`distRaced[index] += curDistRaced`).

## 6. Client Parsing Behavior (snakeoil3_gym)

`ServerState.parse_server_str(...)` in `snakeoil3_gym.py` does:
1. trim trailing whitespace and remove the last character
2. split tuple blocks using `)(`
3. split each tuple by spaces
4. convert scalar/list token strings to numeric values via `destringify(...)`

Result is placed in `ServerState.d`.

Key implication:
- code reading `S = c.S.d` accesses parsed values by field name exactly as sent by server (for example `totalDistFromStart`, not a renamed alias).

## 7. Action Parsing and Restart/Shutdown Tokens

Client action string is parsed server-side using `CarControl`.
Recognized command channels include:
- `accel`, `brake`, `gear`, `steer`, `clutch`, `focus`, `meta`

Special tokens/messages in protocol:
- `***identified***`: handshake confirmation
- `***restart***`: restart notification path
- `***shutdown***`: race/server shutdown notification

In the Python client loop:
- `***restart***` and `***shutdown***` trigger `Client.shutdown()` and loop exit behavior in `get_servers_input()`.

## 8. Practical Accuracy Notes

- The connection is UDP only; there is no persistent session establishment beyond remembering the last `clientAddress` that identified.
- The timeout path does not stop TORCS; it keeps driving with the previous valid controls.
- The server-side ID constant is `SCR-VIS`, but acceptance currently checks only first 3 chars (`SCR`).
- If init parsing fails, track sensor angles silently revert to fixed defaults.
