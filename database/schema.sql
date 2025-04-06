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

-- Markov transitions table
CREATE TABLE IF NOT EXISTS markov_transitions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    current_state TEXT NOT NULL,
    next_state TEXT NOT NULL,
    count INTEGER NOT NULL DEFAULT 0,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(current_state, next_state)
);

-- Create q_learning_backups table
CREATE TABLE IF NOT EXISTS q_learning_backups (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL UNIQUE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create q_learning_rewards table
CREATE TABLE IF NOT EXISTS q_learning_rewards (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    state_hash TEXT NOT NULL,
    action TEXT NOT NULL,
    reward REAL NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (state_hash, action) REFERENCES q_learning_states(state_hash, action) ON DELETE CASCADE
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

-- Indexes for markov_transitions table
CREATE INDEX IF NOT EXISTS idx_markov_state_pair ON markov_transitions(current_state, next_state);
CREATE INDEX IF NOT EXISTS idx_markov_count ON markov_transitions(count);
CREATE INDEX IF NOT EXISTS idx_markov_current_state ON markov_transitions(current_state);
CREATE INDEX IF NOT EXISTS idx_markov_next_state ON markov_transitions(next_state);
CREATE INDEX IF NOT EXISTS idx_markov_updated_at ON markov_transitions(updated_at);

-- Create indexes for q_learning_backups table
CREATE INDEX IF NOT EXISTS idx_q_learning_backups_name ON q_learning_backups(name);
CREATE INDEX IF NOT EXISTS idx_q_learning_backups_created_at ON q_learning_backups(created_at);

-- Create indexes for q_learning_rewards table
CREATE INDEX IF NOT EXISTS idx_q_learning_rewards_state_action ON q_learning_rewards(state_hash, action);
CREATE INDEX IF NOT EXISTS idx_q_learning_rewards_created_at ON q_learning_rewards(created_at);

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

-- Trigger for updating markov_transitions timestamp
CREATE TRIGGER IF NOT EXISTS update_markov_timestamp
    AFTER UPDATE ON markov_transitions
    FOR EACH ROW
    BEGIN
        UPDATE markov_transitions SET updated_at = CURRENT_TIMESTAMP WHERE id = NEW.id;
    END;
