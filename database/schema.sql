-- Games table
CREATE TABLE IF NOT EXISTS games (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    player_name TEXT NOT NULL,
    difficulty TEXT NOT NULL DEFAULT 'medium' CHECK (difficulty IN ('easy', 'medium', 'hard')),
    max_attempts INTEGER NOT NULL DEFAULT 10 CHECK (max_attempts > 0),
    status TEXT NOT NULL DEFAULT 'in_progress' CHECK (status IN ('in_progress', 'completed', 'abandoned')),
    game_score INTEGER DEFAULT 0,
    end_time TIMESTAMP,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Game moves table
CREATE TABLE IF NOT EXISTS game_moves (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    game_id INTEGER NOT NULL,
    word TEXT NOT NULL,
    is_valid BOOLEAN NOT NULL DEFAULT FALSE,
    move_score INTEGER DEFAULT 0,
    feedback TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (game_id) REFERENCES games(id) ON DELETE CASCADE
);

-- Categories table
CREATE TABLE IF NOT EXISTS categories (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL UNIQUE,
    description TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
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

-- Words table
CREATE TABLE IF NOT EXISTS words (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    word TEXT NOT NULL UNIQUE,
    category_id INTEGER,
    domain_id INTEGER,
    frequency INTEGER DEFAULT 0,
    usage_count INTEGER DEFAULT 0,
    allowed BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (category_id) REFERENCES categories(id) ON DELETE SET NULL,
    FOREIGN KEY (domain_id) REFERENCES dictionary_domains(id) ON DELETE SET NULL
);

-- Word usage table
CREATE TABLE IF NOT EXISTS word_usage (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    word_id INTEGER NOT NULL,
    game_id INTEGER NOT NULL,
    score REAL NOT NULL DEFAULT 0.0,
    used_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (word_id) REFERENCES words(id) ON DELETE CASCADE,
    FOREIGN KEY (game_id) REFERENCES games(id) ON DELETE CASCADE
);

-- AI metrics table
CREATE TABLE IF NOT EXISTS ai_metrics (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    game_id INTEGER NOT NULL,
    move_number INTEGER NOT NULL,
    metric_name TEXT NOT NULL,
    value REAL NOT NULL,
    response_time REAL NOT NULL,
    confidence_score REAL NOT NULL CHECK (confidence_score >= 0 AND confidence_score <= 1),
    strategy_used TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (game_id) REFERENCES games(id) ON DELETE CASCADE
);

-- Markov transitions table
CREATE TABLE IF NOT EXISTS markov_transitions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    game_id INTEGER NOT NULL,
    current_state TEXT NOT NULL,
    next_state TEXT NOT NULL,
    count INTEGER NOT NULL DEFAULT 0,
    total_transitions INTEGER NOT NULL DEFAULT 0,
    probability REAL NOT NULL DEFAULT 0.0,
    visit_count INTEGER NOT NULL DEFAULT 0,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (game_id) REFERENCES games(id) ON DELETE CASCADE,
    UNIQUE(game_id, current_state, next_state) ON CONFLICT REPLACE
);

-- Create q_learning_backups table
CREATE TABLE IF NOT EXISTS q_learning_backups (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    game_id INTEGER NOT NULL,
    name TEXT NOT NULL UNIQUE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (game_id) REFERENCES games(id) ON DELETE CASCADE
);

-- Create q_learning_states table
CREATE TABLE IF NOT EXISTS q_learning_states (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    game_id INTEGER NOT NULL,
    state_hash TEXT NOT NULL,
    action TEXT NOT NULL,
    q_value REAL NOT NULL DEFAULT 0.0,
    visit_count INTEGER NOT NULL DEFAULT 1,
    reward REAL NOT NULL DEFAULT 0.0,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (game_id) REFERENCES games(id) ON DELETE CASCADE,
    UNIQUE(game_id, state_hash, action)
);

-- Create q_learning_rewards table
CREATE TABLE IF NOT EXISTS q_learning_rewards (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    game_id INTEGER NOT NULL,
    state_hash TEXT NOT NULL,
    action TEXT NOT NULL,
    reward REAL NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (game_id) REFERENCES games(id) ON DELETE CASCADE,
    FOREIGN KEY (game_id, state_hash, action) REFERENCES q_learning_states(game_id, state_hash, action) ON DELETE CASCADE
);

-- Create q_learning_backup_states table
CREATE TABLE IF NOT EXISTS q_learning_backup_states (
    backup_id INTEGER NOT NULL,
    game_id INTEGER NOT NULL,
    state_hash TEXT NOT NULL,
    action TEXT NOT NULL,
    q_value REAL NOT NULL,
    visit_count INTEGER NOT NULL,
    avg_reward REAL NOT NULL DEFAULT 0.0,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (backup_id, state_hash, action),
    FOREIGN KEY (backup_id) REFERENCES q_learning_backups(id) ON DELETE CASCADE,
    FOREIGN KEY (game_id) REFERENCES games(id) ON DELETE CASCADE
);

-- Naive Bayes tables
CREATE TABLE IF NOT EXISTS naive_bayes_words (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    game_id INTEGER NOT NULL,
    word TEXT NOT NULL,
    probability REAL NOT NULL,
    pattern_type TEXT,
    visit_count INTEGER NOT NULL DEFAULT 1,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (game_id) REFERENCES games(id) ON DELETE CASCADE,
    UNIQUE(game_id, word, pattern_type)
);

-- MCTS tables
CREATE TABLE IF NOT EXISTS mcts_states (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    game_id INTEGER NOT NULL,
    state TEXT NOT NULL,
    visit_count INTEGER NOT NULL DEFAULT 1,
    total_reward REAL NOT NULL DEFAULT 0.0,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (game_id) REFERENCES games(id) ON DELETE CASCADE,
    UNIQUE(game_id, state)
);

CREATE TABLE IF NOT EXISTS mcts_simulations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    game_id INTEGER NOT NULL,
    state TEXT NOT NULL,
    action TEXT NOT NULL,
    visit_count INTEGER NOT NULL DEFAULT 1,
    total_reward REAL NOT NULL DEFAULT 0.0,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (game_id) REFERENCES games(id) ON DELETE CASCADE,
    FOREIGN KEY (game_id, state) REFERENCES mcts_states(game_id, state) ON DELETE CASCADE,
    UNIQUE(game_id, state, action)
);

-- Create indexes for frequently queried columns
CREATE INDEX IF NOT EXISTS idx_games_player_name ON games(player_name);
CREATE INDEX IF NOT EXISTS idx_games_status ON games(status);
CREATE INDEX IF NOT EXISTS idx_games_timestamp ON games(created_at);
CREATE INDEX IF NOT EXISTS idx_game_moves_game_id ON game_moves(game_id);
CREATE INDEX IF NOT EXISTS idx_game_moves_word ON game_moves(word);
CREATE INDEX IF NOT EXISTS idx_words_category_id ON words(category_id);
CREATE INDEX IF NOT EXISTS idx_words_domain_id ON words(domain_id);
CREATE INDEX IF NOT EXISTS idx_words_frequency ON words(frequency);
CREATE INDEX IF NOT EXISTS idx_ai_metrics_game_id ON ai_metrics(game_id);
CREATE INDEX IF NOT EXISTS idx_markov_transitions_game_id ON markov_transitions(game_id);
CREATE INDEX IF NOT EXISTS idx_markov_transitions_current_state ON markov_transitions(current_state);
CREATE INDEX IF NOT EXISTS idx_markov_transitions_next_state ON markov_transitions(next_state);
CREATE INDEX IF NOT EXISTS idx_markov_transitions_probability ON markov_transitions(probability);
CREATE INDEX IF NOT EXISTS idx_q_learning_states_game_id ON q_learning_states(game_id);
CREATE INDEX IF NOT EXISTS idx_q_learning_states_state_hash ON q_learning_states(state_hash);
CREATE INDEX IF NOT EXISTS idx_q_learning_states_action ON q_learning_states(action);
CREATE INDEX IF NOT EXISTS idx_q_learning_states_q_value ON q_learning_states(q_value);
CREATE INDEX IF NOT EXISTS idx_q_learning_rewards_game_id ON q_learning_rewards(game_id);
CREATE INDEX IF NOT EXISTS idx_q_learning_rewards_state_hash ON q_learning_rewards(state_hash);
CREATE INDEX IF NOT EXISTS idx_q_learning_rewards_action ON q_learning_rewards(action);
CREATE INDEX IF NOT EXISTS idx_naive_bayes_words_game_id ON naive_bayes_words(game_id);
CREATE INDEX IF NOT EXISTS idx_naive_bayes_words_word ON naive_bayes_words(word);
CREATE INDEX IF NOT EXISTS idx_naive_bayes_words_pattern_type ON naive_bayes_words(pattern_type);
CREATE INDEX IF NOT EXISTS idx_mcts_states_game_id ON mcts_states(game_id);
CREATE INDEX IF NOT EXISTS idx_mcts_states_state ON mcts_states(state);
CREATE INDEX IF NOT EXISTS idx_mcts_simulations_game_id ON mcts_simulations(game_id);
CREATE INDEX IF NOT EXISTS idx_mcts_simulations_state ON mcts_simulations(state);
CREATE INDEX IF NOT EXISTS idx_mcts_simulations_action ON mcts_simulations(action);

-- Word usage indexes
CREATE INDEX IF NOT EXISTS idx_word_usage_word_id ON word_usage(word_id);
CREATE INDEX IF NOT EXISTS idx_word_usage_game_id ON word_usage(game_id);
CREATE INDEX IF NOT EXISTS idx_word_usage_used_at ON word_usage(used_at);
CREATE INDEX IF NOT EXISTS idx_word_usage_score ON word_usage(score);

-- Create triggers for updating timestamps
CREATE TRIGGER IF NOT EXISTS trg_games_created_at 
AFTER INSERT ON games
BEGIN
    UPDATE games SET created_at = CURRENT_TIMESTAMP WHERE id = NEW.id;
END;

CREATE TRIGGER IF NOT EXISTS trg_games_updated_at 
AFTER UPDATE ON games
BEGIN
    UPDATE games SET updated_at = CURRENT_TIMESTAMP WHERE id = NEW.id;
END;

CREATE TRIGGER IF NOT EXISTS trg_game_moves_created_at 
AFTER INSERT ON game_moves
BEGIN
    UPDATE game_moves SET created_at = CURRENT_TIMESTAMP WHERE id = NEW.id;
END;

CREATE TRIGGER IF NOT EXISTS trg_game_moves_updated_at 
AFTER UPDATE ON game_moves
BEGIN
    UPDATE game_moves SET updated_at = CURRENT_TIMESTAMP WHERE id = NEW.id;
END;

CREATE TRIGGER IF NOT EXISTS trg_categories_created_at 
AFTER INSERT ON categories
BEGIN
    UPDATE categories SET created_at = CURRENT_TIMESTAMP WHERE id = NEW.id;
END;

CREATE TRIGGER IF NOT EXISTS trg_categories_updated_at 
AFTER UPDATE ON categories
BEGIN
    UPDATE categories SET updated_at = CURRENT_TIMESTAMP WHERE id = NEW.id;
END;

CREATE TRIGGER IF NOT EXISTS trg_dictionary_domains_created_at 
AFTER INSERT ON dictionary_domains
BEGIN
    UPDATE dictionary_domains SET created_at = CURRENT_TIMESTAMP WHERE id = NEW.id;
END;

CREATE TRIGGER IF NOT EXISTS trg_dictionary_domains_updated_at 
AFTER UPDATE ON dictionary_domains
BEGIN
    UPDATE dictionary_domains SET updated_at = CURRENT_TIMESTAMP WHERE id = NEW.id;
END;

CREATE TRIGGER IF NOT EXISTS trg_words_created_at 
AFTER INSERT ON words
BEGIN
    UPDATE words SET created_at = CURRENT_TIMESTAMP WHERE id = NEW.id;
END;

CREATE TRIGGER IF NOT EXISTS trg_words_updated_at 
AFTER UPDATE ON words
BEGIN
    UPDATE words SET updated_at = CURRENT_TIMESTAMP WHERE id = NEW.id;
END;

CREATE TRIGGER IF NOT EXISTS trg_ai_metrics_created_at 
AFTER INSERT ON ai_metrics
BEGIN
    UPDATE ai_metrics SET created_at = CURRENT_TIMESTAMP WHERE id = NEW.id;
END;

CREATE TRIGGER IF NOT EXISTS trg_ai_metrics_updated_at 
AFTER UPDATE ON ai_metrics
BEGIN
    UPDATE ai_metrics SET updated_at = CURRENT_TIMESTAMP WHERE id = NEW.id;
END;

CREATE TRIGGER IF NOT EXISTS trg_markov_transitions_created_at 
AFTER INSERT ON markov_transitions
BEGIN
    UPDATE markov_transitions SET created_at = CURRENT_TIMESTAMP WHERE id = NEW.id;
END;

CREATE TRIGGER IF NOT EXISTS trg_markov_transitions_updated_at 
AFTER UPDATE ON markov_transitions
BEGIN
    UPDATE markov_transitions SET updated_at = CURRENT_TIMESTAMP WHERE id = NEW.id;
END;

CREATE TRIGGER IF NOT EXISTS trg_q_learning_states_created_at 
AFTER INSERT ON q_learning_states
BEGIN
    UPDATE q_learning_states SET created_at = CURRENT_TIMESTAMP WHERE id = NEW.id;
END;

CREATE TRIGGER IF NOT EXISTS trg_q_learning_states_updated_at 
AFTER UPDATE ON q_learning_states
BEGIN
    UPDATE q_learning_states SET updated_at = CURRENT_TIMESTAMP WHERE id = NEW.id;
END;

CREATE TRIGGER IF NOT EXISTS trg_q_learning_rewards_created_at 
AFTER INSERT ON q_learning_rewards
BEGIN
    UPDATE q_learning_rewards SET created_at = CURRENT_TIMESTAMP WHERE id = NEW.id;
END;

CREATE TRIGGER IF NOT EXISTS trg_q_learning_rewards_updated_at 
AFTER UPDATE ON q_learning_rewards
BEGIN
    UPDATE q_learning_rewards SET updated_at = CURRENT_TIMESTAMP WHERE id = NEW.id;
END;

CREATE TRIGGER IF NOT EXISTS trg_naive_bayes_words_created_at 
AFTER INSERT ON naive_bayes_words
BEGIN
    UPDATE naive_bayes_words SET created_at = CURRENT_TIMESTAMP WHERE id = NEW.id;
END;

CREATE TRIGGER IF NOT EXISTS trg_naive_bayes_words_updated_at 
AFTER UPDATE ON naive_bayes_words
BEGIN
    UPDATE naive_bayes_words SET updated_at = CURRENT_TIMESTAMP WHERE id = NEW.id;
END;

CREATE TRIGGER IF NOT EXISTS trg_mcts_states_created_at 
AFTER INSERT ON mcts_states
BEGIN
    UPDATE mcts_states SET created_at = CURRENT_TIMESTAMP WHERE id = NEW.id;
END;

CREATE TRIGGER IF NOT EXISTS trg_mcts_states_updated_at 
AFTER UPDATE ON mcts_states
BEGIN
    UPDATE mcts_states SET updated_at = CURRENT_TIMESTAMP WHERE id = NEW.id;
END;

CREATE TRIGGER IF NOT EXISTS trg_mcts_simulations_created_at 
AFTER INSERT ON mcts_simulations
BEGIN
    UPDATE mcts_simulations SET created_at = CURRENT_TIMESTAMP WHERE id = NEW.id;
END;

CREATE TRIGGER IF NOT EXISTS trg_mcts_simulations_updated_at 
AFTER UPDATE ON mcts_simulations
BEGIN
    UPDATE mcts_simulations SET updated_at = CURRENT_TIMESTAMP WHERE id = NEW.id;
END;

-- Create trigger for updating word count in dictionary domains
CREATE TRIGGER IF NOT EXISTS trg_domain_word_count 
AFTER INSERT ON words
BEGIN
    UPDATE dictionary_domains 
    SET word_count = word_count + 1 
    WHERE id = NEW.domain_id;
END;

CREATE TRIGGER IF NOT EXISTS trg_domain_word_count_delete 
AFTER DELETE ON words
BEGIN
    UPDATE dictionary_domains 
    SET word_count = word_count - 1 
    WHERE id = OLD.domain_id;
END;
