#!/usr/bin/env python3
"""
RaspberryPi Camera LED controller
Updated for modern Raspberry Pi OS and Python 3
"""

import os
import fcntl
import struct
import sys

class CameraLED:
    """
    Controller for Raspberry Pi camera LED
    Works with Raspberry Pi 3/4 and newer models
    """
    
    IOCTL_MAILBOX = 0xC0046400
    REQUEST_SUCCESS = 0x80000000
    SET_TAG = 0x38041
    GET_TAG = 0x30041

    def __init__(self, led=134):
        """
        Initialize the camera LED controller
        
        Args:
            led (int): LED number (default: 134 for camera LED)
        """
        try:
            self.__vcio = os.open("/dev/vcio", os.O_RDWR)
            self.__led = led
            self._available = True
        except (OSError, IOError) as e:
            print(f"Warning: Cannot access /dev/vcio: {e}")
            print("Camera LED control not available on this system")
            self._available = False
            self.__vcio = None
    
    def __del__(self):
        """Clean up file descriptor"""
        if hasattr(self, '__vcio') and self.__vcio is not None:
            try:
                os.close(self.__vcio)
            except Exception:
                pass

    def __firmware_request__(self, tag, state=0):
        """
        Send request to Raspberry Pi firmware
        
        Args:
            tag (int): Command tag
            state (int): LED state (0=off, 1=on)
            
        Returns:
            int: New LED state
            
        Raises:
            Exception: If firmware request fails or LED control not available
        """
        if not self._available:
            raise Exception("Camera LED control not available on this system")
        
        try:
            # Create request buffer (8 32-bit integers)
            buffer = struct.pack("=8I",  # format: 8 unsigned 32-bit ints (native endian)
                32,                      # total message length in bytes
                0,                       # request code (0 for request)
                tag,                     # tag identifier
                8,                       # tag data size in bytes
                0,                       # request/response indicator
                self.__led,              # tag data: LED number
                state,                   # tag data: LED state
                0                        # end tag
            )

            # Send request to firmware via ioctl
            result = fcntl.ioctl(self.__vcio, self.IOCTL_MAILBOX, buffer)

            # Unpack response
            (total_len, response_code, resp_tag, resp_size, 
             req_resp_code, led_num, new_state, end_tag) = struct.unpack("=8I", result)

            # Check if request was successful
            if response_code == self.REQUEST_SUCCESS:
                return new_state
            else:
                raise Exception(f"RPi firmware error! Response code: {response_code:08x}")
                
        except (OSError, IOError) as e:
            raise Exception(f"Failed to communicate with firmware: {e}")
        except struct.error as e:
            raise Exception(f"Buffer packing/unpacking error: {e}")

    def on(self):
        """
        Turn the camera LED on
        
        Returns:
            int: New LED state (1 if successful)
            
        Raises:
            Exception: If operation fails
        """
        try:
            return self.__firmware_request__(self.SET_TAG, 1)
        except Exception as e:
            print(f"Error turning LED on: {e}")
            raise

    def off(self):
        """
        Turn the camera LED off
        
        Returns:
            int: New LED state (0 if successful)
            
        Raises:
            Exception: If operation fails
        """
        try:
            return self.__firmware_request__(self.SET_TAG, 0)
        except Exception as e:
            print(f"Error turning LED off: {e}")
            raise

    def toggle(self):
        """
        Toggle the camera LED state
        
        Returns:
            int: New LED state
            
        Raises:
            Exception: If operation fails
        """
        try:
            current_state = self.state()
            if current_state == 1:
                return self.off()
            else:
                return self.on()
        except Exception as e:
            print(f"Error toggling LED: {e}")
            raise

    def state(self):
        """
        Get the current camera LED state
        
        Returns:
            int: Current LED state (0=off, 1=on)
            
        Raises:
            Exception: If operation fails
        """
        try:
            return self.__firmware_request__(self.GET_TAG)
        except Exception as e:
            print(f"Error getting LED state: {e}")
            raise

    def is_available(self):
        """
        Check if camera LED control is available
        
        Returns:
            bool: True if LED control is available
        """
        return self._available


def main():
    """Command line interface for camera LED control"""
    def usage():
        print(f"Usage: {sys.argv[0]} (state|toggle|on|off)")
        print("Commands:")
        print("  state  - Show current LED state")
        print("  toggle - Toggle LED state")
        print("  on     - Turn LED on")
        print("  off    - Turn LED off")
    
    if len(sys.argv) != 2:
        usage()
        sys.exit(1)
    
    try:
        led = CameraLED()
        
        if not led.is_available():
            print("Error: Camera LED control not available on this system")
            sys.exit(1)
        
        command = sys.argv[1].lower()
        
        if command == "state":
            state = led.state()
            print(f"LED State: {state} ({'ON' if state else 'OFF'})")
        elif command == "toggle":
            new_state = led.toggle()
            print(f"LED toggled to: {new_state} ({'ON' if new_state else 'OFF'})")
        elif command == "on":
            new_state = led.on()
            print(f"LED turned on: {new_state} ({'ON' if new_state else 'OFF'})")
        elif command == "off":
            new_state = led.off()
            print(f"LED turned off: {new_state} ({'ON' if new_state else 'OFF'})")
        else:
            print(f"Error: Unknown command '{command}'")
            usage()
            sys.exit(1)
            
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
