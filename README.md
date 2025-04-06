# AI Word Strategy Game

A word-building game featuring AI opponents using various machine learning strategies. Players compete against AI that learns and adapts to gameplay patterns.

## Features

### Core Game
- Word building with shared and private letter pools
- Dynamic scoring system with letter rarity and word length bonuses
- Real-time word validation using Trie-based dictionary
- Turn-based gameplay with time limits

### AI Opponents
- Multiple AI strategies:
  - Q-Learning: Learns optimal word selection through reinforcement learning
  - Naive Bayes: Uses probabilistic word selection based on letter patterns
  - Markov Chain: Predicts word patterns based on letter sequences
  - Monte Carlo Tree Search: Explores possible word combinations
- Adaptive difficulty levels
- Word pattern analysis and learning
- Training system for AI improvement

### Technical Features
- Event-driven architecture
- Modular AI strategy system
- Comprehensive test coverage
- Configurable game parameters
- Logging and debugging tools

## Installation

1. Clone the repository:

git clone https://github.com/yourusername/MC486.github.io.git
cd MC486.github.io


2. Create and activate virtual environment:

python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate


3. Install dependencies:

pip install -r requirements.txt


4. Configure the game:

cp config.yaml.example config.yaml
# Edit config.yaml with your preferred settings


## Usage

### Starting the Game

python game_app.py


### Game Controls
- Enter words using available letters
- Type 'boggle' to request new letters
- Type 'quit' to end the game
- Type 'help' for more commands

### AI Training
To train the AI models:

python -m ai.training.train_models


## Project Structure


.
├── ai/                    # AI components
│   ├── models/           # AI model implementations
│   ├── strategy/         # AI strategy system
│   ├── training/         # AI training utilities
│   └── word_analysis.py  # Word pattern analysis
├── core/                  # Core game components
│   ├── validation/       # Word validation system
│   ├── letter_pool.py    # Letter pool management
│   └── word_scoring.py   # Scoring system
├── engine/               # Game engine
│   ├── game_loop.py      # Main game loop
│   ├── game_state.py     # Game state management
│   └── input_handler.py  # Input processing
├── tests/                # Test suite
│   ├── ai/              # AI component tests
│   ├── core/            # Core component tests
│   └── validation/      # Validation tests
├── utils/               # Utility functions
├── config.yaml          # Game configuration
└── game_app.py          # Main application entry


## Configuration

The game can be configured through `config.yaml`:
- Game settings (word lengths, scoring)
- Letter pool configuration
- AI model parameters
- Dictionary settings
- Logging options

See `config.yaml.example` for all available options.

## Testing

Run the test suite:

pytest


## Development Status

### Completed (Enhancements 1/2)
- Core game mechanics
- Word validation system
- AI model implementations
- Training system
- Event system
- Test coverage

### In Progress (Enhancement 3)
- Database integration
- Game history tracking
- Player statistics
- Scoreboard system
- Database-enabled AI learning

## Acknowledgments

- NLTK for word processing
- Word frequency data sources