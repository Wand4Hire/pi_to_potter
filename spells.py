import os
import subprocess
import time

# Try to import GPIO-related modules
try:
    from gpiozero import LED
    GPIO_AVAILABLE = True
    print("GPIO support available")
except ImportError:
    GPIO_AVAILABLE = False
    print("Warning: gpiozero not available - GPIO functions will be disabled")

# Try to import other modules
try:
    import ble
    BLE_AVAILABLE = True
except ImportError:
    BLE_AVAILABLE = False
    print("Warning: BLE module not available")

try:
    import music
    MUSIC_AVAILABLE = True
except ImportError:
    MUSIC_AVAILABLE = False
    print("Warning: music module not available")

# Initialize GPIO pins if available
digitalLogger = None
otherpin = None

if GPIO_AVAILABLE:
    print("Attempting to initialize GPIOs...")
    try:
        digitalLogger = LED(17)
        otherpin = LED(27)
        print("GPIO pins initialized successfully")
    except Exception as e:
        print(f"Warning: Could not initialize GPIO pins: {e}")
        GPIO_AVAILABLE = False

bubblesSwitch = False


class Spells:
    def __init__(self, args):
        self.args = args
        if hasattr(args, 'use_ble') and args.use_ble and BLE_AVAILABLE:
            try:
                ble.runScanAndSet(False)
            except Exception as e:
                print(f"Warning: Could not initialize BLE: {e}")

    def play_audio(self, filename):
        """Play audio file using available methods"""
        if MUSIC_AVAILABLE:
            try:
                if filename.endswith('.mp3'):
                    music.play_mp3(filename)
                elif filename.endswith('.wav'):
                    music.play_wav(filename)
                else:
                    music.play_mp3(filename)  # Default to mp3
            except Exception as e:
                print(f"Error playing audio with music module: {e}")
                self.fallback_audio(filename)
        else:
            self.fallback_audio(filename)

    def fallback_audio(self, filename):
        """Fallback audio playback using system commands"""
        try:
            if os.path.exists(filename):
                # Try different audio players
                audio_players = ['mpg123', 'aplay', 'paplay', 'omxplayer']
                for player in audio_players:
                    if subprocess.run(['which', player], capture_output=True).returncode == 0:
                        if player == 'mpg123' and filename.endswith('.mp3'):
                            subprocess.Popen([player, filename], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                            break
                        elif player in ['aplay', 'paplay'] and filename.endswith('.wav'):
                            subprocess.Popen([player, filename], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                            break
                        elif player == 'omxplayer':
                            subprocess.Popen([player, filename], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                            break
                else:
                    print(f"No suitable audio player found for {filename}")
            else:
                print(f"Audio file not found: {filename}")
        except Exception as e:
            print(f"Error in fallback audio playback: {e}")

    def run_script(self, script_path):
        """Run shell script if it exists"""
        try:
            if os.path.exists(script_path) and os.access(script_path, os.X_OK):
                subprocess.Popen([script_path], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            else:
                print(f"Script not found or not executable: {script_path}")
        except Exception as e:
            print(f"Error running script {script_path}: {e}")

    def toggle_gpio(self, pin, pin_name):
        """Toggle GPIO pin if available"""
        if GPIO_AVAILABLE and pin is not None:
            try:
                pin.toggle()
                print(f"Toggled {pin_name}")
            except Exception as e:
                print(f"Error toggling {pin_name}: {e}")
        else:
            print(f"GPIO not available - cannot toggle {pin_name}")

    def cast(self, spell):
        global bubblesSwitch, digitalLogger, otherpin
        
        print(f"CAST: {spell}")
        
        # Get home directory path
        home_path = getattr(self.args, 'home', os.path.expanduser('~'))
        music_path = os.path.join(home_path, 'pi_to_potter', 'music')
        
        # Invoke IoT (or any other) actions here
        if spell == "center":
            audio_file = os.path.join(music_path, 'reys.mp3')
            self.play_audio(audio_file)
            
        elif spell == "circle":
            audio_file = os.path.join(music_path, 'audio.mp3')
            self.play_audio(audio_file)
            
        elif spell == "eight":
            print("Toggling digital logger.")
            audio_file = os.path.join(music_path, 'tinkle.mp3')
            self.play_audio(audio_file)
            self.toggle_gpio(digitalLogger, "digital logger")
            
        elif spell == "left":
            print("Toggling magic crystal.")
            if hasattr(self.args, 'use_ble') and self.args.use_ble and BLE_AVAILABLE:
                try:
                    ble.toggleBLE()
                except Exception as e:
                    print(f"Error toggling BLE: {e}")
            else:
                print("BLE not available or not enabled")
                
        elif spell == "square":
            print("Square spell cast")
            # Add your square spell logic here
            
        elif spell == "swish":
            print("Swish spell cast")
            # Add your swish spell logic here
            
        elif spell == "tee":
            print("Toggling bubbles.")
            bubblesSwitch = not bubblesSwitch
            audio_file = os.path.join(music_path, 'spellshot.mp3')
            self.play_audio(audio_file)
            
            if bubblesSwitch:
                script_path = os.path.join(home_path, 'pi_to_potter', 'bubbleson.sh')
                self.run_script(script_path)
            else:
                script_path = os.path.join(home_path, 'pi_to_potter', 'bubblesoff.sh')
                self.run_script(script_path)
                
        elif spell == "triangle":
            print("Toggling outlet.")
            print("Playing audio file...")
            audio_file = os.path.join(music_path, 'wonder.mp3')
            self.play_audio(audio_file)
            
        elif spell == "zee":
            print("Toggling 'other' pin.")
            print("Playing audio file...")
            audio_file = os.path.join(music_path, 'zoo.mp3')
            self.play_audio(audio_file)
            self.toggle_gpio(otherpin, "other pin")
            
        else:
            print(f"Unknown spell: {spell}")
