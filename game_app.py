# game_app.py
# User-facing entry point for the AI word strategy game.

from engine.engine_core import main

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Error starting game: {e}")
        exit(1)
