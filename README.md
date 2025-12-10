# Environmental-analysis-and-distance-measure
This is a script to analyse objects, position and distance. Can also give a detailed description of what it's seeing.

Steps to run this code
A) Installing the Requirements
  # Create a virtual environment named 'venv'
  python -m venv venv

  # Activate it:
  # On Windows:
  venv\Scripts\activate

  # On Mac/Linux:
  source venv/bin/activate

  # Make sure (venv) is showing in terminal
  pip install -r requirements.txt

B) Adding the gemini API key
    #create a new folder alongside the main.py example folder_sample
    #create a sample.env inside folder_sample and in the env file add
    GEMINI_API_KEY = "your_api_key"
    #change these 2 lines in main.py

  load_dotenv(os.path.join(os.path.dirname(__file__), "folder_sample", "sample.env"))
  GEMINI_API_KEY = os.getenv("GEMINI_API_KEY_", "")
