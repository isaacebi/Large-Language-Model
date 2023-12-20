from pathlib import Path

current_directory = Path(__file__).resolve().parent
llm_directory = current_directory.parent

# Now llm_directory points to the 'LLM' directory
print(llm_directory)
