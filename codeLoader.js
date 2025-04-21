const filesToLoad = {
    'game-loop-code': 'engine/game_loop.py',
    'game-state-code': 'engine/game_state.py',
    'engine-core-code': 'engine/engine_core.py',
    'input-handler-code': 'engine/input_handler.py',
    'q-learning-code': 'ai/models/q_learning_model.py',
    'markov-code': 'ai/models/markov_chain.py',
    'mcts-code': 'ai/models/mcts.py',
    'naive-code': 'ai/models/naive_bayes.py',
    'db-manager-code': 'database/manager.py',
    'repo-base-code': 'database/repositories/base_repository.py'
  };
  
  window.addEventListener('DOMContentLoaded', async () => {
    for (const [elementId, path] of Object.entries(filesToLoad)) {
      try {
        const res = await fetch(path);
        if (!res.ok) throw new Error(`Could not fetch ${path}`);
        const text = await res.text();
        document.getElementById(elementId).textContent = text;
      } catch (err) {
        document.getElementById(elementId).textContent = `// Failed to load ${path}\n// ${err.message}`;
      }
    }
  });
  
  // Toggle collapsible blocks
  function toggleBlock(id) {
    const el = document.getElementById(id);
    el.style.display = el.style.display === 'block' ? 'none' : 'block';
  }
  
  // Tab switching logic
  function switchTab(tabName) {
    document.querySelectorAll('.tab-content').forEach(el => el.style.display = 'none');
    document.getElementById(tabName).style.display = 'block';
  }
  