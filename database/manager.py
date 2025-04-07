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
        self.create_tables()
        
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
            # Only try to get results if the query returns any
            if cursor.description:
                columns = [description[0] for description in cursor.description]
                return [dict(zip(columns, row)) for row in cursor.fetchall()]
            return []
            
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
        """Create all necessary database tables if they don't exist."""
        try:
            # Create word usage table
            self.execute("""
                CREATE TABLE IF NOT EXISTS word_usage (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    word TEXT UNIQUE NOT NULL,
                    allowed BOOLEAN NOT NULL,
                    num_played INTEGER DEFAULT 0,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Create categories table
            self.execute("""
                CREATE TABLE IF NOT EXISTS categories (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT UNIQUE NOT NULL,
                    description TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Create naive_bayes table
            self.execute("""
                CREATE TABLE IF NOT EXISTS naive_bayes (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    word TEXT NOT NULL,
                    category TEXT NOT NULL,
                    probability REAL NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(word, category)
                )
            """)
            
            # Create mcts table
            self.execute("""
                CREATE TABLE IF NOT EXISTS mcts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    state TEXT NOT NULL,
                    action TEXT NOT NULL,
                    visits INTEGER DEFAULT 0,
                    value REAL DEFAULT 0,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(state, action)
                )
            """)
            
            # Create q_learning table
            self.execute("""
                CREATE TABLE IF NOT EXISTS q_learning (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    state TEXT NOT NULL,
                    action TEXT NOT NULL,
                    q_value REAL DEFAULT 0,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(state, action)
                )
            """)
            
            # Create markov_chain table
            self.execute("""
                CREATE TABLE IF NOT EXISTS markov_chain (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    current_state TEXT NOT NULL,
                    next_state TEXT NOT NULL,
                    probability REAL NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(current_state, next_state)
                )
            """)
            
            # Create games table
            self.execute("""
                CREATE TABLE IF NOT EXISTS games (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    start_time TIMESTAMP NOT NULL,
                    end_time TIMESTAMP,
                    score INTEGER DEFAULT 0,
                    words_played TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            self.logger.info("Database tables created successfully")
        except Exception as e:
            self.logger.error(f"Error creating database tables: {str(e)}")
            raise
            
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
        
        # Q-Learning indexes
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_q_learning_state ON q_learning_states(state_hash)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_q_learning_action ON q_learning_states(action)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_q_learning_updated ON q_learning_states(last_updated)")
        
        # Markov indexes
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_markov_current ON markov_transitions(current_state)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_markov_next ON markov_transitions(next_state)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_markov_updated ON markov_transitions(last_updated)")
        
        # MCTS indexes
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_mcts_state ON mcts_states(state_hash)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_mcts_action ON mcts_states(action)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_mcts_updated ON mcts_states(last_updated)")
        
        # Naive Bayes indexes
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_bayes_word ON naive_bayes_probabilities(word)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_bayes_pattern ON naive_bayes_probabilities(pattern_type)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_bayes_updated ON naive_bayes_probabilities(last_updated)")
        
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

    def get_cursor(self):
        """
        Get a database cursor.
        
        Returns:
            sqlite3.Cursor: Database cursor
        """
        return self.conn.cursor()

    def commit(self):
        """
        Commit the current transaction.
        """
        self.conn.commit()

    def get_mcts_repository(self):
        """
        Get the MCTS repository.
        
        Returns:
            MCTSRepository: MCTS repository instance
        """
        from database.repositories.mcts_repository import MCTSRepository
        return MCTSRepository(self)

    def get_q_learning_repository(self):
        """
        Get the Q-Learning repository.
        
        Returns:
            QLearningRepository: Q-Learning repository instance
        """
        from .repositories.q_learning_repository import QLearningRepository
        return QLearningRepository(self)

    def get_naive_bayes_repository(self):
        """Get the Naive Bayes repository instance."""
        from .repositories.naive_bayes_repository import NaiveBayesRepository
        return NaiveBayesRepository(self)