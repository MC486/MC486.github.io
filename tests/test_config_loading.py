# tests/test_config_loading.py
# Unit test to ensure the config.yaml file loads correctly and has required sections.

import unittest
import yaml

class TestConfigYAML(unittest.TestCase):
    def test_config_yaml_valid(self):
        """
        Ensures that config.yaml is present, can be loaded, and contains required top-level keys.
        """
        try:
            with open("config.yaml", "r") as f:
                config = yaml.safe_load(f) # Load the YAML configuration from config.yaml.

            # Check for top-level sections expected in the configuration
            self.assertIn('game', config, "Missing 'game' section in config.yaml") # Check if 'game' section exists.
            self.assertIn('letter_pool', config, "Missing 'letter_pool' section in config.yaml") # Check if 'letter_pool' section exists.
            self.assertIn('dictionary', config, "Missing 'dictionary' section in config.yaml") # Check if 'dictionary' section exists.

        except FileNotFoundError:
            self.fail("config.yaml not found. Make sure it exists at the root level.") # Fail the test if config.yaml is not found.
        except yaml.YAMLError as e:
            self.fail(f"config.yaml contains invalid YAML: {e}") # Fail the test if config.yaml contains invalid YAML.
        except Exception as e:
            self.fail(f"Unexpected error while loading config.yaml: {e}") # Fail the test for any other unexpected exception.

if __name__ == "__main__":
    unittest.main() # Run the unit tests.