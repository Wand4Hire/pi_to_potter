#!/usr/bin/env python3
"""
Audio playback module for Raspberry Pi Potter project
Updated for modern Raspberry Pi OS with better compatibility
"""

import subprocess
import sys
import traceback
import os
import signal
import threading
import time
from pathlib import Path

# Global process holders
wav_process = None
mp3_process = None
process_lock = threading.Lock()

# Available audio players (in order of preference)
AUDIO_PLAYERS = {
    'wav': ['aplay', 'paplay', 'sox', 'ffplay'],
    'mp3': ['mpg123', 'mpg321', 'ffplay', 'cvlc']
}


def find_audio_player(file_type):
    """
    Find available audio player for specified file type
    
    Args:
        file_type (str): 'wav' or 'mp3'
        
    Returns:
        str or None: Path to available audio player
    """
    players = AUDIO_PLAYERS.get(file_type, [])
    
    for player in players:
        try:
            # Check if player is available
            result = subprocess.run(['which', player], 
                                  capture_output=True, 
                                  text=True, 
                                  timeout=5)
            if result.returncode == 0:
                return player.strip()
        except (subprocess.TimeoutExpired, subprocess.SubprocessError):
            continue
    
    return None


def kill_process_safely(process):
    """
    Safely terminate a subprocess
    
    Args:
        process: subprocess.Popen object
    """
    if process is None:
        return
    
    try:
        if process.poll() is None:  # Process is still running
            # Try SIGTERM first
            process.terminate()
            try:
                process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                # Force kill if SIGTERM didn't work
                process.kill()
                process.wait()
    except Exception as e:
        print(f"Error killing process: {e}")


def play_wav(file_path):
    """
    Play WAV audio file
    
    Args:
        file_path (str): Path to WAV file
    """
    global wav_process
    
    if not os.path.exists(file_path):
        print(f"WAV file not found: {file_path}")
        return
    
    try:
        with process_lock:
            # Stop any existing WAV playback
            if wav_process is not None:
                kill_process_safely(wav_process)
                wav_process = None
            
            # Find available audio player
            player = find_audio_player('wav')
            if not player:
                print("No suitable WAV player found")
                return
            
            # Start playback
            print(f"Playing WAV: {file_path} using {player}")
            
            if player == 'aplay':
                wav_process = subprocess.Popen(
                    [player, file_path],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL
                )
            elif player == 'paplay':
                wav_process = subprocess.Popen(
                    [player, file_path],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL
                )
            elif player == 'sox':
                wav_process = subprocess.Popen(
                    [player, file_path, '-d'],  # -d for default output
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL
                )
            elif player == 'ffplay':
                wav_process = subprocess.Popen(
                    [player, '-nodisp', '-autoexit', file_path],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL
                )
            
    except Exception as e:
        print(f"Error playing WAV file: {e}")
        traceback.print_exc()


def stop_wav():
    """Stop WAV audio playback"""
    global wav_process
    
    try:
        with process_lock:
            if wav_process is not None:
                print("Stopping WAV playback")
                kill_process_safely(wav_process)
                wav_process = None
    except Exception as e:
        print(f"Error stopping WAV: {e}")
        traceback.print_exc()


def play_mp3(file_path):
    """
    Play MP3 audio file
    
    Args:
        file_path (str): Path to MP3 file
    """
    global mp3_process
    
    if not os.path.exists(file_path):
        print(f"MP3 file not found: {file_path}")
        return
    
    try:
        with process_lock:
            # Stop any existing MP3 playback
            stop_mp3()
            
            # Find available audio player
            player = find_audio_player('mp3')
            if not player:
                print("No suitable MP3 player found")
                return
            
            # Start playback
            print(f"Playing MP3: {file_path} using {player}")
            
            if player in ['mpg123', 'mpg321']:
                mp3_process = subprocess.Popen(
                    [player, file_path],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL
                )
            elif player == 'ffplay':
                mp3_process = subprocess.Popen(
                    [player, '-nodisp', '-autoexit', file_path],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL
                )
            elif player == 'cvlc':
                mp3_process = subprocess.Popen(
                    [player, '--intf', 'dummy', '--play-and-exit', file_path],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL
                )
            
    except Exception as e:
        print(f"Error playing MP3 file: {e}")
        traceback.print_exc()


def stop_mp3():
    """Stop MP3 audio playback"""
    global mp3_process
    
    try:
        with process_lock:
            # Kill any existing MP3 processes by name (legacy method)
            for player in ['mpg123', 'mpg321', 'ffplay', 'cvlc']:
                try:
                    subprocess.run(['killall', player], 
                                 capture_output=True, 
                                 timeout=2)
                except (subprocess.TimeoutExpired, subprocess.SubprocessError):
                    pass
            
            # Kill our specific process
            if mp3_process is not None:
                print("Stopping MP3 playback")
                kill_process_safely(mp3_process)
                mp3_process = None
                
    except Exception as e:
        print(f"Error stopping MP3: {e}")
        traceback.print_exc()


def stop_all():
    """Stop all audio playback"""
    stop_wav()
    stop_mp3()


def play_audio(file_path):
    """
    Play audio file (auto-detect format)
    
    Args:
        file_path (str): Path to audio file
    """
    if not os.path.exists(file_path):
        print(f"Audio file not found: {file_path}")
        return
    
    file_ext = Path(file_path).suffix.lower()
    
    if file_ext == '.wav':
        play_wav(file_path)
    elif file_ext == '.mp3':
        play_mp3(file_path)
    else:
        print(f"Unsupported audio format: {file_ext}")


def test_audio_system():
    """Test if audio system is working"""
    print("Testing audio system...")
    
    # Test WAV player
    wav_player = find_audio_player('wav')
    if wav_player:
        print(f"WAV player available: {wav_player}")
    else:
        print("No WAV player found")
    
    # Test MP3 player
    mp3_player = find_audio_player('mp3')
    if mp3_player:
        print(f"MP3 player available: {mp3_player}")
    else:
        print("No MP3 player found")
    
    return wav_player is not None or mp3_player is not None


# Cleanup function for graceful shutdown
def cleanup():
    """Clean up any running audio processes"""
    stop_all()


# Register cleanup function
import atexit
atexit.register(cleanup)


if __name__ == "__main__":
    # Test the audio system
    if test_audio_system():
        print("Audio system is functional")
    else:
        print("No audio players found - audio will not work")
