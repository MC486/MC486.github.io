from database.manager import DatabaseManager
import os

def main():
    # Ensure data directory exists
    os.makedirs('data', exist_ok=True)
    
    # Initialize database
    db = DatabaseManager('data/game.db')
    print("Database initialized successfully!")

if __name__ == "__main__":
    main() 