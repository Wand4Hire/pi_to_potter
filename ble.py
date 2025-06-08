import time
import os
from pathlib import Path

# Try to import BLE libraries
try:
    from bluepy import btle
    from bluepy.btle import Scanner, DefaultDelegate
    BLUEPY_AVAILABLE = True
    print("bluepy BLE support available")
except ImportError:
    BLUEPY_AVAILABLE = False
    print("Warning: bluepy not available - BLE functions will be disabled")

# Try to import music module
try:
    import music
    MUSIC_AVAILABLE = True
except ImportError:
    MUSIC_AVAILABLE = False
    print("Warning: music module not available")

found = False
failures = 0
scanner = None

# Initialize scanner if bluepy is available
if BLUEPY_AVAILABLE:
    class ScanDelegate(DefaultDelegate):
        def __init__(self):
            DefaultDelegate.__init__(self)

        def handleDiscovery(self, dev, isNewDev, isNewData):
            global found
            if isNewDev:
                print("Discovered device", dev.addr)
                if dev.addr == 'cb:22:99:ce:97:8f':
                    found = True
            elif isNewData:
                print("Received new data from", dev.addr)

    try:
        scanner = Scanner().withDelegate(ScanDelegate())
    except Exception as e:
        print(f"Warning: Could not initialize BLE scanner: {e}")
        BLUEPY_AVAILABLE = False


def runScanAndSet(state):
    """Scan for BLE device and set its state"""
    global found, failures, scanner
    
    if not BLUEPY_AVAILABLE:
        print("BLE not available - skipping scan")
        return
    
    found = False
    peripheral = None
    
    try:
        # Scan for devices
        devices = scanner.scan(3.0)
        
        # Try to connect to the specific device
        peripheral = btle.Peripheral('cb:22:99:ce:97:8f', btle.ADDR_TYPE_RANDOM)
        failures = 0
        
        # Get the characteristic
        guid = '713d0003503e4c75ba943148f18d941e'
        characteristics = peripheral.getCharacteristics(uuid=guid)
        
        if not characteristics:
            print("Warning: Characteristic not found")
            return
            
        characteristic = characteristics[0]
        
        # Set the state
        if state:
            turnOn(characteristic)
            turnOn(characteristic)  # Double call as in original
        else:
            turnOff(characteristic)
            turnOff(characteristic)  # Double call as in original
            
    except btle.BTLEDisconnectError as e:
        print(f"BLE disconnection error: {e}")
        failures += 1
        if failures < 10:
            print(f"Retrying... (attempt {failures}/10)")
            time.sleep(1)
            runScanAndSet(state)
        else:
            print("Max retries reached, giving up")
            failures = 0
    except btle.BTLEException as e:
        print(f"BLE error: {e}")
        failures += 1
        if failures < 10:
            print(f"Retrying... (attempt {failures}/10)")
            time.sleep(1)
            runScanAndSet(state)
        else:
            print("Max retries reached, giving up")
            failures = 0
    except Exception as e:
        print(f"Unexpected error in BLE operation: {e}")
        failures += 1
        if failures < 10:
            print(f"Retrying... (attempt {failures}/10)")
            time.sleep(1)
            runScanAndSet(state)
        else:
            print("Max retries reached, giving up")
            failures = 0
    finally:
        # Always disconnect
        if peripheral is not None:
            try:
                peripheral.disconnect()
            except Exception as e:
                print(f"Error disconnecting: {e}")


def turnOn(characteristic):
    """Turn on the BLE device"""
    try:
        # Set Output
        command = bytearray([0x53, 0x04, 0x01])  # S, 4, 1
        print(f"Sending command: {command}")
        characteristic.write(command)

        # Turn on
        command = bytearray([0x54, 0x04, 0x01])  # T, 4, 1
        print(f"Sending command: {command}")
        characteristic.write(command)
    except Exception as e:
        print(f"Error in turnOn: {e}")


def turnOff(characteristic):
    """Turn off the BLE device"""
    try:
        # Set Output
        command = bytearray([0x53, 0x04, 0x01])  # S, 4, 1
        print(f"Sending command: {command}")
        characteristic.write(command)

        # Turn off
        command = bytearray([0x54, 0x04, 0x00])  # T, 4, 0
        print(f"Sending command: {command}")
        characteristic.write(command)
    except Exception as e:
        print(f"Error in turnOff: {e}")


bleState = False


def toggleBLE():
    """Toggle the BLE device state"""
    global bleState
    
    if not BLUEPY_AVAILABLE:
        print("BLE not available - cannot toggle")
        return
    
    # Get home directory
    home_address = str(Path.home())
    bell_file = os.path.join(home_address, 'pi_to_potter', 'music', 'bell.wav')
    
    # Play bell sound if available
    if MUSIC_AVAILABLE:
        try:
            music.play_wav(bell_file)
        except Exception as e:
            print(f"Error playing bell sound: {e}")
    
    # Toggle state
    bleState = not bleState
    print(f"Toggling BLE to: {'ON' if bleState else 'OFF'}")
    
    # Set the new state
    runScanAndSet(bleState)
    
    # Wait and stop sound
    time.sleep(10)
    
    if MUSIC_AVAILABLE:
        try:
            music.stop_wav()
        except Exception as e:
            print(f"Error stopping bell sound: {e}")


def is_ble_available():
    """Check if BLE functionality is available"""
    return BLUEPY_AVAILABLE


def get_ble_state():
    """Get current BLE state"""
    return bleState
