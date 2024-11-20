import os
import sys

models = {}

# Use a relative path for the models directory
models_directory = os.path.dirname(__file__)

if models_directory not in sys.path:
    sys.path.append(models_directory)

# Load models dynamically
for filename in os.listdir(models_directory):
    if filename.endswith('.py') and filename != '__init__.py':
        module_name = filename[:-3]
        try:
            module = __import__(module_name)

            for item in dir(module):
                attr = getattr(module, item)
                if isinstance(attr, type) and item.endswith('Model'):
                    model_instance = attr()  # Create an instance of the model class
                    model = model_instance.get_model()
                    parameters = model_instance.get_params()
                    test_size = model_instance.get_test_size()
                    random_state = model_instance.get_random_state()

                    # Store model information in the models dictionary
                    models[attr.__name__] = {
                        "class": attr,
                        "parameters": parameters,
                        "test_size": test_size,
                        "random_state": random_state
                    }
                    break

        except Exception as e:
            print(f"Error loading model from {filename}: {str(e)}")

if __name__ == "__main__":
    print("Loaded models:")
    for model_name, model_info in models.items():
        print(f"{model_name}: {model_info}")
