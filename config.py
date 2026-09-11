"""config.py — Runtime configuration for the IIT Pokerbots 2026 engine.

This module centralises all environment-level knobs required to launch a local
two-bot game via ``engine.py``.  Adjust ``BOT_1_FILE`` and ``BOT_2_FILE`` to
point at the bots you wish to pit against each other, and set ``PYTHON_CMD``
to match your local Python interpreter alias.

Attributes:
    PYTHON_CMD (str): Shell command used to invoke Python.  On most Linux/macOS
        systems the interpreter is ``'python3'``; on Windows it is typically
        ``'python'``.  Change this if the engine fails to spawn bot sub-processes.
    BOT_1_NAME (str): Human-readable display name for Bot 1, shown in log output.
    BOT_1_FILE (str): Path to the Bot 1 entry-point script relative to the
        project root (e.g. ``'./bot.py'``).
    BOT_2_NAME (str): Human-readable display name for Bot 2.
    BOT_2_FILE (str): Path to the Bot 2 entry-point script.
    GAME_LOG_FOLDER (str): Directory where round-by-round game logs are written.
        Created automatically if it does not exist.
"""

PYTHON_CMD = "python"
# For linux and mac, the code python cmd is sometimes 'python3' instead of 'python'

BOT_1_NAME = 'BotA'
BOT_1_FILE = './bot.py'

BOT_2_NAME = 'BotB'
BOT_2_FILE = './bot.py'

# GAME PROGRESS IS RECORDED HERE
GAME_LOG_FOLDER = './logs'