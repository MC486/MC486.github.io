# AI Word Strategy Game

A turn-based word-building game where players compete against AI opponents trained on real dictionary data and evolving gameplay patterns. Developed as a capstone enhancement project to demonstrate software design, AI integration, and long-term extensibility.

---

## 🎮 Features

### 🧠 Core Game
- Strategic word creation using shared and private letter pools
- Dynamic scoring system with:
  - Word length bonuses
  - Letter rarity multipliers
  - Repeat-word fatigue penalties
- Clean CLI interface for fast-paced turn-based play
- Real-time word validation using Trie-backed dictionary

### 🤖 AI Opponents
- Multiple AI models:
  - **Q-Learning** – Reinforcement agent learns long-term reward patterns
  - **Naive Bayes** – Predicts player words using letter-based probabilities
  - **Markov Chain** – Generates plausible letter transitions
  - **Monte Carlo Tree Search** – Simulates possible move sequences
- Weighted ensemble prediction with model selection via Q-learning
- Adaptive behavior based on opponent style

### 💾 Database Integration
- SQLite-based persistent storage
- Player profile and scoring history
- AI model training data persistence
- Performance analytics and leaderboards

### 🛠 Technical Highlights
- Modular game engine architecture
- Fully tested game loop, state, input, scoring, and letter generation
- `config.yaml` for tunable gameplay settings
- Structured logs for debugging and gameplay review
- Clean Git branching and feature isolation per enhancement

---

## ⚙️ Installation

```
git clone https://github.com/yourusername/MC486.github.io.git
cd MC486.github.io
```

Create and activate a virtual environment:

```
python -m venv .venv
.venv\Scripts\activate      # Windows
# or
source .venv/bin/activate   # macOS/Linux
```

Install dependencies:

```
pip install -r requirements.txt
```

---

## 🧩 Usage

Start the game:

```
python main.py
```

### In-Game Commands
- Type a word using available letters
- `boggle` — Redraw private letters
- `quit` — Exit the game
- `help` — Show full command list
- `stats` — View your gameplay statistics

---

## 🗃 Project Structure

```
.
├── ai/
│   ├── markov_model.py
│   ├── mcts_model.py
│   ├── naive_bayes_model.py
│   ├── q_learning_model.py
│   └── trie_validator.py
├── core/
│   ├── letter_pool.py
│   ├── word_scoring.py
│   └── player.py
├── engine/
│   ├── game_loop.py
│   ├── game_state.py
│   └── input_handler.py
├── tests/
│   ├── test_game_loop.py
│   ├── test_game_state.py
│   ├── test_input_handler.py
│   ├── test_letter_pool.py
│   ├── test_word_list_loader.py
│   └── test_word_scoring.py
├── utils/
│   └── word_list_loader.py
├── database/
│   ├── schema.sql
│   ├── db.py
│   └── db_helpers.py
├── config.yaml
├── requirements.txt
└── main.py
```

---

## 🧪 Testing

Run the full test suite:

```
pytest
```

All modules are tested including game state handling, letter generation logic, scoring algorithms, input parsing, AI models, and database operations.

---

## 🧬 Configuration

Edit `config.yaml` to customize:
- Letter pool size and distribution
- Scoring rules and fatigue settings
- AI prediction time budget (for MCTS)
- Logging verbosity and format
- Database connection parameters

---

## 📈 Development Status

### ✅ Enhancement 1: Software Design & Engineering
- ✅ Modular architecture and class hierarchy
- ✅ Letter generation logic
- ✅ Scoring and repeat-word fatigue
- ✅ Logging, testing, and config system

### ✅ Enhancement 2: Algorithms & ML Integration
- ✅ AI model architecture implementation
- ✅ Markov, MCTS, Q-learning, Naive Bayes modules completed
- ✅ Prediction pipeline and ensemble model

### ✅ Enhancement 3: Database & Persistence
- ✅ SQLite integration
- ✅ AI learning from gameplay history
- ✅ Player analytics and scoreboards

---
