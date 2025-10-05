import sys
import types
import importlib
import unittest


# Stub external dependencies before importing the module under test
if 'flask' not in sys.modules:
    flask_stub = types.ModuleType('flask')

    class DummyFlask:
        def __init__(self, *args, **kwargs):
            pass

        def route(self, *args, **kwargs):
            def decorator(func):
                return func

            return decorator

        def run(self, *args, **kwargs):
            pass

    flask_stub.Flask = DummyFlask
    flask_stub.request = types.SimpleNamespace(json=None)
    flask_stub.jsonify = lambda *args, **kwargs: None
    flask_stub.send_from_directory = lambda *args, **kwargs: None
    sys.modules['flask'] = flask_stub

if 'flask_cors' not in sys.modules:
    flask_cors_stub = types.ModuleType('flask_cors')
    flask_cors_stub.CORS = lambda *args, **kwargs: None
    sys.modules['flask_cors'] = flask_cors_stub

if 'openai' not in sys.modules:
    openai_stub = types.ModuleType('openai')

    class DummyClient:
        def __init__(self, *args, **kwargs):
            pass

    openai_stub.OpenAI = DummyClient
    sys.modules['openai'] = openai_stub

main_with_llm = importlib.import_module('src.main_with_llm')


class ELMUpdateTests(unittest.TestCase):
    def test_llm_guided_update_uses_hidden_activations(self):
        model = main_with_llm.LLMGuidedTowerDefenseELM(input_size=2, hidden_size=1, output_size=2, random_state=0)
        model.learning_rate = 0.1
        model.input_weights = [[0.4], [-0.2]]
        model.hidden_bias = [0.0]
        model.output_weights = [[0.1, -0.2]]
        x = [2.0, -3.0]
        target = [1.0, 0.0]

        before_update = [row[:] for row in model.output_weights]

        # Expected hidden activation based on normalized inputs
        x_norm = [val / abs(val) if abs(val) > 1e-8 else val for val in x]
        hidden = []
        for j in range(len(model.hidden_bias)):
            sum_val = model.hidden_bias[j]
            for i in range(len(x_norm)):
                sum_val += x_norm[i] * model.input_weights[i][j]
            hidden.append(model.tanh(sum_val))

        # Forward pass for expected prediction
        prediction = []
        for j in range(len(model.output_weights[0])):
            sum_val = 0.0
            for i in range(len(hidden)):
                sum_val += hidden[i] * model.output_weights[i][j]
            prediction.append(model.sigmoid(sum_val))

        error = [target[i] - prediction[i] for i in range(len(target))]
        expected_updates = [
            [model.learning_rate * error[j] * hidden[i] for j in range(len(model.output_weights[i]))]
            for i in range(len(model.output_weights))
        ]

        model.update(x, target)

        for i in range(len(model.output_weights)):
            for j in range(len(model.output_weights[i])):
                expected = before_update[i][j] + expected_updates[i][j]
                self.assertAlmostEqual(model.output_weights[i][j], expected, places=7)

    def test_simple_update_uses_hidden_activations(self):
        model = main_with_llm.SimpleTowerDefenseELM(input_size=2, hidden_size=1, output_size=2, random_state=0)
        model.learning_rate = 0.05
        model.input_weights = [[-0.3], [0.25]]
        model.hidden_bias = [0.1]
        model.output_weights = [[0.05, 0.15]]
        x = [-4.0, 5.0]
        target = [0.0, 1.0]

        before_update = [row[:] for row in model.output_weights]

        x_norm = [val / abs(val) if abs(val) > 1e-8 else val for val in x]
        hidden = []
        for j in range(len(model.hidden_bias)):
            sum_val = model.hidden_bias[j]
            for i in range(len(x_norm)):
                sum_val += x_norm[i] * model.input_weights[i][j]
            hidden.append(model.tanh(sum_val))

        prediction = []
        for j in range(len(model.output_weights[0])):
            sum_val = 0.0
            for i in range(len(hidden)):
                sum_val += hidden[i] * model.output_weights[i][j]
            prediction.append(model.sigmoid(sum_val))

        error = [target[i] - prediction[i] for i in range(len(target))]
        expected_updates = [
            [model.learning_rate * error[j] * hidden[i] for j in range(len(model.output_weights[i]))]
            for i in range(len(model.output_weights))
        ]

        model.update(x, target)

        for i in range(len(model.output_weights)):
            for j in range(len(model.output_weights[i])):
                expected = before_update[i][j] + expected_updates[i][j]
                self.assertAlmostEqual(model.output_weights[i][j], expected, places=7)


if __name__ == '__main__':
    unittest.main()
