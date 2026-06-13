# Testing Documentation for IBM Fast Car AI

This document provides instructions on how to run and interpret tests for the TORCS Reinforcement Learning project.

## Overview

The testing suite is divided into three main categories:
1.  **Unit Tests**: Validate individual components (parsing, utils, normalization).
2.  **Integration Tests**: Verify interactions between components using mocks (client-server connection, environment-client interaction).
3.  **Functional Tests**: Test high-level features like vision mode and track selection.

## Prerequisites

Ensure you have the virtual environment activated and dependencies installed:

```bash
# Activate virtual environment
source .venv/bin/activate

# Ensure dependencies are installed
pip install -e .
```

## Running Tests

### 1. Run All Tests
To run all available tests in the `tests/` directory:

```bash
python3 -m unittest discover tests
```

### 2. Run Specific Test Files
If you want to focus on a specific area:

*   **SCR State Parsing**: `python3 -m unittest tests/test_scr_state.py`
*   **Utility Functions**: `python3 -m unittest tests/test_scr_utils.py`
*   **RL Environment Logic**: `python3 -m unittest tests/test_rl_env.py`
*   **Functional Features**: `python3 -m unittest tests/test_functional.py`
*   **Process Management**: `python3 -m unittest tests/test_process_manager.py`
*   **Client Connection**: `python3 -m unittest tests/test_client_connection.py`

## Test Categories Explained

### Unit Tests
*   `test_scr_state.py`: Verifies that the UDP messages from the TORCS server are correctly parsed into Python dictionaries and that driver actions are correctly encoded into strings.
*   `test_scr_utils.py`: Tests helper functions like `clip` and `destringify`.

### Integration Tests
*   `test_rl_env.py`: Uses a `MagicMock` for the TORCS backend to test reward calculations, termination conditions, and observation normalization without needing to launch the actual TORCS game.
*   `test_client_connection.py`: Uses a `MockTorcsServer` (a background UDP thread) to verify that the `Client` class can correctly handshake with a server.
*   `test_process_manager.py`: Mocks `subprocess.run` to verify that the code correctly identifies UDP port owners and issues the correct kill commands for different platforms.

### Functional Tests
*   `test_functional.py`:
    *   **Vision Mode**: Verifies that enabling `vision=True` correctly expands the observation space to include flattened image data.
    *   **Race Config**: Ensures that custom XML track configurations are correctly passed down to the launcher.

## Troubleshooting

### Common Issues
*   **ModuleNotFoundError: No module named 'numpy'**: Ensure you are running the tests inside the virtual environment (`source .venv/bin/activate`).
*   **Socket Errors**: Some tests (like `test_client_connection.py`) bind to local ports. If the port (default 3002) is already in use, the test may fail.
*   **ResourceWarnings**: You might see `ResourceWarning: unclosed <socket.socket...>`. These are expected in tests that intentionally disrupt connections to test timeout/reconnect logic.

