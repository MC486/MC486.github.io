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
            
            # AI Domain Tables
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS markov_transitions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    current_state TEXT NOT NULL,
                    next_state TEXT NOT NULL,
                    probability REAL NOT NULL,
                    count INTEGER DEFAULT 1,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS q_learning_states (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    state_hash TEXT NOT NULL UNIQUE,
                    value REAL NOT NULL,
                    visit_count INTEGER DEFAULT 0,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Game Domain Tables
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS game_sessions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    player_id INTEGER NOT NULL,
                    start_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    end_time TIMESTAMP,
                    score INTEGER DEFAULT 0,
                    status TEXT DEFAULT 'active'
                )
            """)
            
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS game_turns (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id INTEGER REFERENCES game_sessions(id),
                    turn_number INTEGER NOT NULL,
                    word TEXT,
                    score INTEGER,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Dictionary Domain Tables
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS words (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    word TEXT NOT NULL UNIQUE,
                    category_id INTEGER,
                    frequency INTEGER DEFAULT 0,
                    confidence REAL DEFAULT 1.0
                )
            """)
            
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS categories (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL UNIQUE,
                    description TEXT
                )
            """)
            
            # Metrics Domain Tables
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS performance_metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    metric_name TEXT NOT NULL,
                    value REAL NOT NULL,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    context TEXT
                )
            """)
            
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS system_errors (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    error_type TEXT NOT NULL,
                    message TEXT NOT NULL,
                    stack_trace TEXT,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    context TEXT
                )
            """)
            
            # Create indexes
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_markov_current_state ON markov_transitions(current_state)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_q_learning_state_hash ON q_learning_states(state_hash)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_game_sessions_player ON game_sessions(player_id)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_words_category ON words(category_id)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_performance_metrics_name ON performance_metrics(metric_name)")