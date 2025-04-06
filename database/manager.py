# src/database/manager.py
from typing import Optional, Any
from sqlite3 import Connection, Cursor, connect
from contextlib import contextmanager
import logging

class DatabaseManager:
    def __init__(self, db_path: str):
        """Initialize the database manager with the SQLite database path."""
        self.db_path = db_path
        self.logger = logging.getLogger(__name__)
        
    def __enter__(self):
        """Enter the context manager."""
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit the context manager."""
        pass
        
    @contextmanager
    def get_connection(self) -> Connection:
        """Get a database connection with automatic cleanup."""
        conn = connect(self.db_path)
        try:
            yield conn
            conn.commit()
        except Exception as e:
            conn.rollback()
            self.logger.error(f"Database error: {str(e)}")
            raise
        finally:
            conn.close()
            
    def execute_query(self, query: str, params: Optional[tuple] = None) -> list:
        """Execute a query and return the results as a list of dictionaries."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(query, params or ())
            columns = [description[0] for description in cursor.description]
            return [dict(zip(columns, row)) for row in cursor.fetchall()]
            
    def execute(self, query: str, params: Optional[tuple] = None) -> None:
        """Execute a query that doesn't return results."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(query, params or ())
            
    def execute_many(self, query: str, params_list: list[tuple]) -> None:
        """Execute a query multiple times with different parameters."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.executemany(query, params_list)
            
    def get_one(self, query: str, params: Optional[tuple] = None) -> Optional[dict]:
        """Execute a query and return a single row as a dictionary."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(query, params or ())
            row = cursor.fetchone()
            if row:
                columns = [description[0] for description in cursor.description]
                return dict(zip(columns, row))
            return None
            
    def get_scalar(self, query: str, params: Optional[tuple] = None) -> Optional[Any]:
        """Execute a query and return a single scalar value."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(query, params or ())
            row = cursor.fetchone()
            return row[0] if row else None
            
    def table_exists(self, table_name: str) -> bool:
        """Check if a table exists in the database."""
        query = """
            SELECT name FROM sqlite_master 
            WHERE type='table' AND name=?
        """
        return bool(self.get_scalar(query, (table_name,)))
            
    def drop_tables(self) -> None:
        """Drop all tables in the database."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            # Get all table names
            cursor.execute("""
                SELECT name FROM sqlite_master 
                WHERE type='table' AND name NOT LIKE 'sqlite_%'
            """)
            tables = [row[0] for row in cursor.fetchall()]
            
            # Drop each table
            for table in tables:
                cursor.execute(f"DROP TABLE IF EXISTS {table}")
                
    def create_tables(self) -> None:
        """Create all necessary database tables."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            # Create games table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS games (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    player_name TEXT NOT NULL,
                    difficulty TEXT NOT NULL,
                    max_attempts INTEGER NOT NULL,
                    status TEXT NOT NULL DEFAULT 'in_progress',
                    score INTEGER,
                    start_time TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    end_time TIMESTAMP,
                    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    CHECK (difficulty IN ('easy', 'medium', 'hard')),
                    CHECK (status IN ('in_progress', 'completed', 'abandoned')),
                    CHECK (max_attempts > 0)
                )
            """)
            
            # Create game moves table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS game_moves (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    game_id INTEGER NOT NULL,
                    word TEXT NOT NULL,
                    is_valid BOOLEAN NOT NULL,
                    feedback TEXT NOT NULL,
                    timestamp TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (game_id) REFERENCES games(id) ON DELETE CASCADE
                )
            """)
            
            # Create words table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS words (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    word TEXT NOT NULL UNIQUE,
                    category_id INTEGER,
                    frequency INTEGER DEFAULT 0,
                    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (category_id) REFERENCES categories(id) ON DELETE SET NULL
                )
            """)
            
            # Create categories table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS categories (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL UNIQUE,
                    description TEXT,
                    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Create AI metrics table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS ai_metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    game_id INTEGER NOT NULL,
                    move_number INTEGER NOT NULL,
                    response_time REAL NOT NULL,
                    confidence_score REAL NOT NULL,
                    strategy_used TEXT NOT NULL,
                    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (game_id) REFERENCES games(id) ON DELETE CASCADE
                )
            """)
            
            # Create dictionary domains table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS dictionary_domains (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL UNIQUE,
                    description TEXT,
                    word_count INTEGER DEFAULT 0,
                    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Create indexes
            self._create_indexes(cursor)
            
            # Create triggers
            self._create_triggers(cursor)
            
            conn.commit()
            
    def _create_indexes(self, cursor: Cursor) -> None:
        """Create necessary indexes."""
        # Games table indexes
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_games_player_name ON games(player_name)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_games_status ON games(status)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_games_difficulty ON games(difficulty)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_games_start_time ON games(start_time)")
        
        # Game moves indexes
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_game_moves_game_id ON game_moves(game_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_game_moves_timestamp ON game_moves(timestamp)")
        
        # Words table indexes
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_words_word ON words(word)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_words_category ON words(category_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_words_frequency ON words(frequency)")
        
        # Categories table indexes
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_categories_name ON categories(name)")
        
        # AI metrics indexes
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_ai_metrics_game ON ai_metrics(game_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_ai_metrics_time ON ai_metrics(created_at)")
        
        # Dictionary domains indexes
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_domains_name ON dictionary_domains(name)")
        
    def _create_triggers(self, cursor: Cursor) -> None:
        """Create necessary triggers."""
        # Update timestamps for games
        cursor.execute("""
            CREATE TRIGGER IF NOT EXISTS update_games_timestamp
            AFTER UPDATE ON games
            BEGIN
                UPDATE games SET updated_at = CURRENT_TIMESTAMP WHERE id = NEW.id;
            END;
        """)
        
        # Update timestamps for words
        cursor.execute("""
            CREATE TRIGGER IF NOT EXISTS update_words_timestamp
            AFTER UPDATE ON words
            BEGIN
                UPDATE words SET updated_at = CURRENT_TIMESTAMP WHERE id = NEW.id;
            END;
        """)
        
        # Update timestamps for categories
        cursor.execute("""
            CREATE TRIGGER IF NOT EXISTS update_categories_timestamp
            AFTER UPDATE ON categories
            BEGIN
                UPDATE categories SET updated_at = CURRENT_TIMESTAMP WHERE id = NEW.id;
            END;
        """)
        
        # Update word count in dictionary domains
        cursor.execute("""
            CREATE TRIGGER IF NOT EXISTS update_domain_word_count
            AFTER INSERT ON words
            BEGIN
                UPDATE dictionary_domains 
                SET word_count = word_count + 1,
                    updated_at = CURRENT_TIMESTAMP
                WHERE id = NEW.domain_id;
            END;
        """)