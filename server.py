#!/usr/bin/env python3
"""
Updated HTTP Server for Raspberry Pi OS with OpenCV support
Compatible with Python 3.x and modern Raspberry Pi OS
"""

from http.server import BaseHTTPRequestHandler, HTTPServer
import json
import logging
from spells import Spells

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class SpellHandler(BaseHTTPRequestHandler):
    """HTTP request handler for spell casting endpoints"""
    
    def __init__(self, *args, **kwargs):
        # Initialize spells instance
        self.spells = Spells()
        super().__init__(*args, **kwargs)
    
    def do_GET(self):
        """Handle GET requests for spell casting"""
        try:
            # Set response headers
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')  # Enable CORS if needed
            self.end_headers()
            
            # Route handling with error checking
            spell_routes = {
                "/circle": "circle",
                "/square": "square", 
                "/zee": "zee",
                "/eight": "eight",
                "/triangle": "triangle",  # Fixed: was missing .cast()
                "/tee": "tee",
                "/left": "left",
                "/center": "center"
            }
            
            if self.path in spell_routes:
                spell_name = spell_routes[self.path]
                logger.info(f"Casting spell: {spell_name}")
                
                # Cast the spell
                result = self.spells.cast(spell_name)
                
                # Send success response
                response = {"done": True, "spell": spell_name, "result": result}
                self.wfile.write(json.dumps(response).encode('utf-8'))
                
            elif self.path == "/":
                # Basic status endpoint
                response = {"status": "Server running", "available_spells": list(spell_routes.keys())}
                self.wfile.write(json.dumps(response).encode('utf-8'))
                
            else:
                # Handle unknown paths
                self.send_error(404, f"Unknown spell path: {self.path}")
                
        except Exception as e:
            logger.error(f"Error processing request {self.path}: {str(e)}")
            self.send_error(500, f"Internal server error: {str(e)}")
    
    def do_POST(self):
        """Handle POST requests for more complex spell operations"""
        try:
            content_length = int(self.headers.get('Content-Length', 0))
            post_data = self.rfile.read(content_length)
            
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            
            # Parse JSON data
            try:
                data = json.loads(post_data.decode('utf-8'))
                spell_name = data.get('spell')
                parameters = data.get('parameters', {})
                
                if spell_name:
                    logger.info(f"Casting spell via POST: {spell_name} with params: {parameters}")
                    result = self.spells.cast(spell_name, **parameters)
                    response = {"done": True, "spell": spell_name, "result": result}
                else:
                    response = {"done": False, "error": "No spell specified"}
                    
            except json.JSONDecodeError:
                response = {"done": False, "error": "Invalid JSON data"}
                
            self.wfile.write(json.dumps(response).encode('utf-8'))
            
        except Exception as e:
            logger.error(f"Error processing POST request: {str(e)}")
            self.send_error(500, f"Internal server error: {str(e)}")
    
    def log_message(self, format, *args):
        """Override to use our logger instead of stderr"""
        logger.info(f"{self.address_string()} - {format % args}")


def run_server(host='0.0.0.0', port=8000):
    """
    Run the HTTP server
    
    Args:
        host (str): Host to bind to (default: '0.0.0.0' for all interfaces)
        port (int): Port to listen on (default: 8000)
    """
    try:
        # Create server instance
        server = HTTPServer((host, port), SpellHandler)
        logger.info(f"Started HTTP server on {host}:{port}")
        logger.info("Available endpoints:")
        logger.info("  GET  /circle, /square, /zee, /eight, /triangle, /tee, /left, /center")
        logger.info("  POST / (with JSON: {'spell': 'name', 'parameters': {}})")
        logger.info("  GET  / (status)")
        
        # Start serving
        server.serve_forever()
        
    except KeyboardInterrupt:
        logger.info("Received Ctrl+C, shutting down server...")
        server.socket.close()
        logger.info("Server stopped")
        
    except OSError as e:
        if e.errno == 98:  # Address already in use
            logger.error(f"Port {port} is already in use. Try a different port.")
        else:
            logger.error(f"OS Error: {str(e)}")
            
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        server.socket.close()


if __name__ == "__main__":
    import argparse
    
    # Command line argument parsing
    parser = argparse.ArgumentParser(description='Spell Casting HTTP Server for Raspberry Pi')
    parser.add_argument('--host', default='0.0.0.0', help='Host to bind to (default: 0.0.0.0)')
    parser.add_argument('--port', type=int, default=8000, help='Port to listen on (default: 8000)')
    parser.add_argument('--debug', action='store_true', help='Enable debug logging')
    
    args = parser.parse_args()
    
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
    
    run_server(args.host, args.port)
