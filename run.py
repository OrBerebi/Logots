import sys
import os
import warnings
from dotenv import load_dotenv

warnings.filterwarnings("ignore", message=".*swapaxes.*", category=FutureWarning)

load_dotenv()  # loads .env into os.environ

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))

if __name__ == "__main__":
    try:
        from logots.gui.gui_controller import main
        
        print("🤖 Starting Logots Robot Controller...")
        
        # 3. Launch the App
        main()
        
    except Exception as e:
        print(f"❌ Runtime Error: {e}")