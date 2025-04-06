-- Games table
CREATE TABLE IF NOT EXISTS games (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    player_name TEXT NOT NULL,
    difficulty TEXT NOT NULL CHECK (difficulty IN ('easy', 'medium', 'hard')),
    max_attempts INTEGER NOT NULL CHECK (max_attempts > 0),
    status TEXT NOT NULL DEFAULT 'in_progress' CHECK (status IN ('in_progress', 'completed', 'abandoned')),
    score INTEGER,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Game moves table
CREATE TABLE IF NOT EXISTS game_moves (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    game_id INTEGER NOT NULL,
    word TEXT NOT NULL,
    is_valid BOOLEAN NOT NULL,
    feedback TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (game_id) REFERENCES games(id) ON DELETE CASCADE
);

-- Words table
CREATE TABLE IF NOT EXISTS words (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    word TEXT NOT NULL UNIQUE,
    category_id INTEGER,
    frequency INTEGER DEFAULT 0,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (category_id) REFERENCES categories(id) ON DELETE SET NULL
);

-- Categories table
CREATE TABLE IF NOT EXISTS categories (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL UNIQUE,
    description TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- AI metrics table
CREATE TABLE IF NOT EXISTS ai_metrics (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    game_id INTEGER NOT NULL,
    move_number INTEGER NOT NULL,
    response_time REAL NOT NULL,
    confidence_score REAL NOT NULL CHECK (confidence_score >= 0 AND confidence_score <= 1),
    strategy_used TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (game_id) REFERENCES games(id) ON DELETE CASCADE
);

-- Dictionary domains table
CREATE TABLE IF NOT EXISTS dictionary_domains (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL UNIQUE,
    description TEXT,
    word_count INTEGER DEFAULT 0,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Indexes for games table
CREATE INDEX IF NOT EXISTS idx_games_player_name ON games(player_name);
CREATE INDEX IF NOT EXISTS idx_games_status ON games(status);
CREATE INDEX IF NOT EXISTS idx_games_difficulty ON games(difficulty);
CREATE INDEX IF NOT EXISTS idx_games_start_time ON games(created_at);

-- Indexes for game_moves table
CREATE INDEX IF NOT EXISTS idx_game_moves_game_id ON game_moves(game_id);
CREATE INDEX IF NOT EXISTS idx_game_moves_timestamp ON game_moves(created_at);

-- Indexes for words table
CREATE INDEX IF NOT EXISTS idx_words_category ON words(category_id);
CREATE INDEX IF NOT EXISTS idx_words_frequency ON words(frequency);

-- Indexes for categories table
CREATE INDEX IF NOT EXISTS idx_categories_name ON categories(name);

-- Indexes for AI metrics table
CREATE INDEX IF NOT EXISTS idx_ai_metrics_game_id ON ai_metrics(game_id);
CREATE INDEX IF NOT EXISTS idx_ai_metrics_move_number ON ai_metrics(move_number);

-- Indexes for dictionary domains table
CREATE INDEX IF NOT EXISTS idx_domains_name ON dictionary_domains(name);
CREATE INDEX IF NOT EXISTS idx_domains_word_count ON dictionary_domains(word_count);

-- Trigger to update updated_at timestamp
CREATE TRIGGER IF NOT EXISTS update_games_timestamp
AFTER UPDATE ON games
FOR EACH ROW
BEGIN
    UPDATE games SET updated_at = CURRENT_TIMESTAMP WHERE id = NEW.id;
END;

CREATE TRIGGER IF NOT EXISTS update_words_timestamp
AFTER UPDATE ON words
FOR EACH ROW
BEGIN
    UPDATE words SET updated_at = CURRENT_TIMESTAMP WHERE id = NEW.id;
END;

CREATE TRIGGER IF NOT EXISTS update_categories_timestamp
AFTER UPDATE ON categories
FOR EACH ROW
BEGIN
    UPDATE categories SET updated_at = CURRENT_TIMESTAMP WHERE id = NEW.id;
END;

CREATE TRIGGER IF NOT EXISTS update_domains_timestamp
AFTER UPDATE ON dictionary_domains
FOR EACH ROW
BEGIN
    UPDATE dictionary_domains SET updated_at = CURRENT_TIMESTAMP WHERE id = NEW.id;
END;

-- Trigger for updating domain word count
CREATE TRIGGER IF NOT EXISTS update_domain_word_count
AFTER INSERT ON words
FOR EACH ROW
BEGIN
    UPDATE dictionary_domains 
    SET word_count = word_count + 1 
    WHERE id = NEW.domain_id;
END;

-- Trigger for updating domain word count on delete
CREATE TRIGGER IF NOT EXISTS update_domain_word_count_delete
AFTER DELETE ON words
FOR EACH ROW
BEGIN
    UPDATE dictionary_domains 
    SET word_count = word_count - 1 
    WHERE id = OLD.domain_id;
END;

-- Trigger for updating domain word count on update
CREATE TRIGGER IF NOT EXISTS update_domain_word_count_update
AFTER UPDATE OF domain_id ON words
FOR EACH ROW
BEGIN
    -- Decrease count for old domain
    UPDATE dictionary_domains 
    SET word_count = word_count - 1 
    WHERE id = OLD.domain_id;
    
    -- Increase count for new domain
    UPDATE dictionary_domains 
    SET word_count = word_count + 1 
    WHERE id = NEW.domain_id;
END;
