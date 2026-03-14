import traceback
try:
    from src.graph import app
    print("Successfully imported app from src.graph!")
except Exception as e:
    traceback.print_exc()
