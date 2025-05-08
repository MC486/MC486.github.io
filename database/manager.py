# src/database/manager.py
from typing import Optional, Any
from sqlite3 import Connection, Cursor, connect
from contextlib import contextmanager
import logging
import os

class DatabaseManager:
    def __init__(self, db_path: str):
        """Initialize the database manager with the SQLite database path."""
        self.db_path = db_path
        self.logger = logging.getLogger(__name__)
        self.conn = None
        self.initialize_database()
        
    def __enter__(self):
        """Enter the context manager."""
        if not self.conn:
            self.conn = connect(self.db_path)
            self.conn.execute("PRAGMA foreign_keys = ON")
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit the context manager."""
        if self.conn:
            if exc_type is not None:
                self.conn.rollback()
            else:
                self.conn.commit()
            self.conn.close()
            self.conn = None
            
    @contextmanager
    def get_connection(self) -> Connection:
        """Get a database connection with automatic cleanup.
        
        Yields:
            sqlite3.Connection: A database connection that will be automatically closed.
        """
        if self.conn:
            yield self.conn
        else:
            conn = connect(self.db_path)
            try:
                conn.execute("PRAGMA foreign_keys = ON")
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
        if not self.conn:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(query, params or ())
                if cursor.description:
                    columns = [description[0] for description in cursor.description]
                    return [dict(zip(columns, row)) for row in cursor.fetchall()]
                if query.strip().upper().startswith('INSERT'):
                    return [{'id': cursor.lastrowid}]
                return []
        else:
            cursor = self.conn.cursor()
            cursor.execute(query, params or ())
            if cursor.description:
                columns = [description[0] for description in cursor.description]
                return [dict(zip(columns, row)) for row in cursor.fetchall()]
            if query.strip().upper().startswith('INSERT'):
                return [{'id': cursor.lastrowid}]
            return []
            
    def execute(self, query: str, params: Optional[tuple] = None) -> Optional[int]:
        """Execute a query that doesn't return results."""
        if not self.conn:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(query, params or ())
                if query.strip().upper().startswith('INSERT'):
                    if 'RETURNING' in query.upper():
                        row = cursor.fetchone()
                        return row[0] if row else None
                    return cursor.lastrowid
                return None
        else:
            cursor = self.conn.cursor()
            cursor.execute(query, params or ())
            if query.strip().upper().startswith('INSERT'):
                if 'RETURNING' in query.upper():
                    row = cursor.fetchone()
                    return row[0] if row else None
                return cursor.lastrowid
            return None
            
    def execute_many(self, query: str, params_list: list[tuple]) -> None:
        """Execute a query multiple times with different parameters."""
        if not self.conn:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.executemany(query, params_list)
        else:
            cursor = self.conn.cursor()
            cursor.executemany(query, params_list)
            
    def execute_script(self, script: str) -> None:
        """Execute a SQL script containing multiple statements."""
        if not self.conn:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.executescript(script)
        else:
            cursor = self.conn.cursor()
            cursor.executescript(script)
            
    def get_one(self, query: str, params: Optional[tuple] = None) -> Optional[dict]:
        """Execute a query and return a single row as a dictionary."""
        if not self.conn:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(query, params or ())
                row = cursor.fetchone()
                if row:
                    columns = [description[0] for description in cursor.description]
                    return dict(zip(columns, row))
                return None
        else:
            cursor = self.conn.cursor()
            cursor.execute(query, params or ())
            row = cursor.fetchone()
            if row:
                columns = [description[0] for description in cursor.description]
                return dict(zip(columns, row))
            return None
            
    def get_scalar(self, query: str, params: Optional[tuple] = None) -> Optional[Any]:
        """Execute a query and return a single scalar value."""
        if not self.conn:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(query, params or ())
                row = cursor.fetchone()
                return row[0] if row else None
        else:
            cursor = self.conn.cursor()
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
        if not self.conn:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT name FROM sqlite_master 
                    WHERE type='table' AND name NOT LIKE 'sqlite_%'
                """)
                tables = [row[0] for row in cursor.fetchall()]
                for table in tables:
                    cursor.execute(f"DROP TABLE IF EXISTS {table}")
        else:
            cursor = self.conn.cursor()
            cursor.execute("""
                SELECT name FROM sqlite_master 
                WHERE type='table' AND name NOT LIKE 'sqlite_%'
            """)
            tables = [row[0] for row in cursor.fetchall()]
            for table in tables:
                cursor.execute(f"DROP TABLE IF EXISTS {table}")
                
    def create_tables(self) -> None:
        """Create all necessary database tables if they don't exist."""
        try:
            if not self.conn:
                with self.get_connection() as conn:
                    cursor = conn.cursor()
                    cursor.execute("PRAGMA foreign_keys = ON")
                    self.execute_schema_file()
            else:
                cursor = self.conn.cursor()
                cursor.execute("PRAGMA foreign_keys = ON")
                self.execute_schema_file()
            self.logger.info("Database tables created successfully")
        except Exception as e:
            self.logger.error(f"Error creating tables: {str(e)}")
            raise

    def execute_schema_file(self) -> None:
        """Execute the schema.sql file to create all tables, indexes, and triggers."""
        try:
            schema_path = os.path.join(os.path.dirname(__file__), 'schema.sql')
            with open(schema_path, 'r') as f:
                schema_sql = f.read()

            if not self.conn:
                with self.get_connection() as conn:
                    cursor = conn.cursor()
                    cursor.execute("PRAGMA foreign_keys = ON")
                    cursor.executescript(schema_sql)
            else:
                cursor = self.conn.cursor()
                cursor.execute("PRAGMA foreign_keys = ON")
                cursor.executescript(schema_sql)
            self.logger.info("Schema file executed successfully")
        except Exception as e:
            self.logger.error(f"Error executing schema file: {str(e)}")
            raise

    def initialize_database(self) -> None:
        """Initialize the database by creating tables and setting up the schema."""
        try:
            # Ensure the database directory exists
            db_dir = os.path.dirname(self.db_path)
            if db_dir and not os.path.exists(db_dir):
                os.makedirs(db_dir)
                
            # Create the database file if it doesn't exist
            if not os.path.exists(self.db_path):
                with self.get_connection() as conn:
                    pass

            # Start a new connection for the initialization process
            with self.get_connection() as conn:
                cursor = conn.cursor()
                
                # Enable foreign keys
                cursor.execute("PRAGMA foreign_keys = OFF")
                
                # Drop all existing tables
                cursor.execute("""
                    SELECT name FROM sqlite_master 
                    WHERE type='table' AND name NOT LIKE 'sqlite_%'
                """)
                tables = [row[0] for row in cursor.fetchall()]
                for table in tables:
                    cursor.execute(f"DROP TABLE IF EXISTS {table}")
                
                # Execute the schema file to create tables
                schema_path = os.path.join(os.path.dirname(__file__), 'schema.sql')
                with open(schema_path, 'r') as f:
                    schema_sql = f.read()
                cursor.executescript(schema_sql)
                
                # Re-enable foreign keys
                cursor.execute("PRAGMA foreign_keys = ON")
            
            self.logger.info("Database initialized successfully")
        except Exception as e:
            self.logger.error(f"Error initializing database: {str(e)}")
            raise
            
    def get_mcts_repository(self):
        """Get the MCTS repository instance."""
        from .repositories.mcts_repository import MCTSRepository
        return MCTSRepository(self)
        
    def get_q_learning_repository(self):
        """Get the Q-Learning repository instance."""
        from .repositories.q_learning_repository import QLearningRepository
        return QLearningRepository(self)
        
    def get_naive_bayes_repository(self):
        """Get the Naive Bayes repository instance."""
        from .repositories.naive_bayes_repository import NaiveBayesRepository
        return NaiveBayesRepository(self)
        
    def get_markov_repository(self, game_id: int):
        """Get the Markov Chain repository instance."""
        from .repositories.markov_repository import MarkovRepository
        return MarkovRepository(self, game_id)
        
    def get_word_repository(self):
        """Get the Word repository instance."""
        from .repositories.word_repository import WordRepository
        return WordRepository(self)
        
    def get_category_repository(self):
        """Get the Category repository instance."""
        from .repositories.category_repository import CategoryRepository
        return CategoryRepository(self)
        
    def get_game_repository(self):
        """Get the Game repository instance."""
        from .repositories.game_repository import GameRepository
        return GameRepository(self)
        
    def get_word_usage_repository(self):
        """Get the Word Usage repository instance."""
        from .repositories.word_usage_repository import WordUsageRepository
        return WordUsageRepository(self)
        
    def close(self):
        """Close the database connection."""
        if self.conn:
            self.conn.close()
            self.conn = None