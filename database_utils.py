"""
database_utils.py - Shared database utilities for the Procurement RAG system
"""

import os
import sqlite3
import pandas as pd
import logging
from contextlib import contextmanager
from typing import Optional, List, Dict, Any
from constants import DB_PATH, CSV_PATH, VENDOR_COL, COST_COL, DESC_COL, COMMODITY_COL

logger = logging.getLogger(__name__)

class DatabaseManager:
    """Singleton database manager for connection pooling and utilities"""
    
    _instance = None
    _connection = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(DatabaseManager, cls).__new__(cls)
        return cls._instance
    
    def __init__(self):
        """Initialize database manager"""
        if not hasattr(self, 'initialized'):
            self.db_path = DB_PATH
            self.ensure_database_exists()
            self.initialized = True
    
    def ensure_database_exists(self):
        """Ensure database exists, create if not"""
        # Create data directory if it doesn't exist
        data_dir = os.path.dirname(self.db_path)
        if data_dir and not os.path.exists(data_dir):
            os.makedirs(data_dir, exist_ok=True)
            logger.info(f"Created data directory: {data_dir}")
        
        # Check if database exists
        if not os.path.exists(self.db_path):
            logger.warning(f"Database not found at {self.db_path}")
            self.create_database_from_csv()
    
    def create_database_from_csv(self):
        """Create database from CSV file"""
        if not os.path.exists(CSV_PATH):
            logger.error(f"CSV file not found: {CSV_PATH}")
            raise FileNotFoundError(f"CSV file required: {CSV_PATH}")
        
        try:
            logger.info(f"Creating database from {CSV_PATH}")
            
            # Read CSV
            df = pd.read_csv(CSV_PATH, low_memory=False)
            logger.info(f"Loaded {len(df)} records from CSV")
            
            # Clean column names (replace spaces with underscores, etc.)
            original_columns = df.columns
            df.columns = df.columns.str.strip().str.replace(' ', '_').str.replace(r'[^a-zA-Z0-9_]', '', regex=True)
            renamed_columns = {orig: new for orig, new in zip(original_columns, df.columns) if orig != new}
            if renamed_columns:
                logger.info(f"Cleaned column names: {renamed_columns}")

            # Create database
            conn = sqlite3.connect(self.db_path)
            
            # Save to database
            df.to_sql('procurement', conn, if_exists='replace', index=False)
            
            # Create indexes for performance
            cursor = conn.cursor()
            cursor.execute(f"CREATE INDEX IF NOT EXISTS idx_vendor ON procurement({VENDOR_COL})")
            cursor.execute(f"CREATE INDEX IF NOT EXISTS idx_cost ON procurement({COST_COL})")
            cursor.execute(f"CREATE INDEX IF NOT EXISTS idx_commodity ON procurement({COMMODITY_COL})")
            
            # Add full-text search if possible
            try:
                cursor.execute(f"""
                CREATE VIRTUAL TABLE IF NOT EXISTS procurement_fts 
                USING fts5({VENDOR_COL}, {DESC_COL}, {COMMODITY_COL})
                """)
                cursor.execute(f"""
                INSERT INTO procurement_fts 
                SELECT {VENDOR_COL}, {DESC_COL}, {COMMODITY_COL} 
                FROM procurement
                """)
            except sqlite3.OperationalError:
                logger.warning("FTS5 not available - full-text search disabled")
            
            conn.commit()
            conn.close()
            
            logger.info(f"Database created successfully at {self.db_path}")
            
        except Exception as e:
            logger.error(f"Failed to create database: {e}")
            raise
    
    @contextmanager
    def get_connection(self):
        """Get database connection as context manager"""
        conn = None
        try:
            conn = sqlite3.connect(self.db_path, timeout=30.0)
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA cache_size=10000")
            conn.execute("PRAGMA synchronous=NORMAL")
            yield conn
        except sqlite3.Error as e:
            logger.error(f"Database connection error: {e}")
            if conn:
                conn.rollback()
            raise
        finally:
            if conn:
                conn.close()
    
    def execute_query(self, query: str, params: Optional[List] = None) -> pd.DataFrame:
        """Execute query and return DataFrame"""
        with self.get_connection() as conn:
            if params is None:
                params = []
            return pd.read_sql_query(query, conn, params=params)
    
    def execute_scalar(self, query: str, params: Optional[List] = None) -> Any:
        """Execute query and return scalar result"""
        df = self.execute_query(query, params)
        if not df.empty:
            return df.iloc[0, 0]
        return None
    
    def get_stats(self) -> Dict[str, Any]:
        """Get database statistics"""
        stats = {}
        
        try:
            # Total records
            stats['total_records'] = self.execute_scalar(
                "SELECT COUNT(*) FROM procurement"
            )
            
            # Unique vendors
            stats['unique_vendors'] = self.execute_scalar(
                f"SELECT COUNT(DISTINCT {VENDOR_COL}) FROM procurement"
            )
            
            # Total spending
            stats['total_spending'] = self.execute_scalar(
                f"SELECT SUM(CAST({COST_COL} AS FLOAT)) FROM procurement WHERE {COST_COL} IS NOT NULL"
            )
            
            # Average order
            stats['average_order'] = self.execute_scalar(
                f"SELECT AVG(CAST({COST_COL} AS FLOAT)) FROM procurement WHERE {COST_COL} IS NOT NULL"
            )
            
            # Database size
            stats['database_size_mb'] = os.path.getsize(self.db_path) / (1024 * 1024)
            
        except Exception as e:
            logger.error(f"Failed to get database stats: {e}")
            stats['error'] = str(e)
        
        return stats
    
    def vendor_exists(self, vendor_name: str) -> bool:
        """Check if vendor exists in database"""
        count = self.execute_scalar(
            f"SELECT COUNT(*) FROM procurement WHERE UPPER({VENDOR_COL}) LIKE ?",
            [f"%{vendor_name.upper()}%"]
        )
        return count > 0
    
    def get_vendor_list(self, pattern: Optional[str] = None, limit: int = 100) -> List[str]:
        """Get list of vendors matching pattern"""
        if pattern:
            query = f"""
            SELECT DISTINCT {VENDOR_COL} 
            FROM procurement 
            WHERE UPPER({VENDOR_COL}) LIKE ?
            ORDER BY {VENDOR_COL}
            LIMIT ?
            """
            params = [f"%{pattern.upper()}%", limit]
        else:
            query = f"""
            SELECT DISTINCT {VENDOR_COL} 
            FROM procurement 
            WHERE {VENDOR_COL} IS NOT NULL
            ORDER BY {VENDOR_COL}
            LIMIT ?
            """
            params = [limit]
        
        df = self.execute_query(query, params)
        return df[VENDOR_COL].tolist()
    
    def search_full_text(self, search_term: str, limit: int = 20) -> pd.DataFrame:
        """Full-text search if FTS5 is available"""
        try:
            query = """
            SELECT p.*, fts.rank
            FROM procurement p
            JOIN procurement_fts fts ON 
                p.rowid = fts.rowid
            WHERE procurement_fts MATCH ?
            ORDER BY rank
            LIMIT ?
            """
            return self.execute_query(query, [search_term, limit])
        except sqlite3.OperationalError:
            # Fallback to LIKE queries
            logger.debug("FTS not available, using LIKE fallback")
            pattern = f"%{search_term}%"
            query = f"""
            SELECT *
            FROM procurement
            WHERE UPPER({VENDOR_COL}) LIKE ? 
               OR UPPER({DESC_COL}) LIKE ?
               OR UPPER({COMMODITY_COL}) LIKE ?
            LIMIT ?
            """
            return self.execute_query(query, [pattern.upper()] * 3 + [limit])

# Singleton instance
db_manager = DatabaseManager()

# Utility functions for backward compatibility
def get_db_connection():
    """Get database connection (for backward compatibility)"""
    return db_manager.get_connection()

def safe_execute_query(query: str, params: Optional[List] = None) -> pd.DataFrame:
    """Execute query safely with parameters"""
    return db_manager.execute_query(query, params)

def ensure_database_exists():
    """Ensure database exists"""
    db_manager.ensure_database_exists()

def get_database_stats() -> Dict[str, Any]:
    """Get database statistics"""
    return db_manager.get_stats()