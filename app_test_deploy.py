#!/usr/bin/env python3
"""
Tower Defense ELM - Test Deploy Version
"""

from flask import Flask

app = Flask(__name__)

@app.route('/')
def index():
    """Test deployment page"""
    return """
<!DOCTYPE html>
<html>
<head>
    <title>Deploy Test</title>
</head>
<body>
    <h1>Deploy Test - Version Check</h1>
    <p>If you see this, the new deployment is working!</p>
    <div id="apiKeyInput">API Key Input Test</div>
    <div id="gameOverOverlay">Game Over Overlay Test</div>
    <script>
        console.log('New version deployed successfully!');
        console.log('API Key Input:', document.getElementById('apiKeyInput'));
        console.log('Game Over Overlay:', document.getElementById('gameOverOverlay'));
    </script>
</body>
</html>
    """

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
