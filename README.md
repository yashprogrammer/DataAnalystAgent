# Titanic Data Analysis Agents

This project demonstrates a multi-agent system for performing data analysis on the Titanic dataset using advanced AI agents and tools. The main script, `be_with_humanPaid.py`, orchestrates a team of specialized agents to analyze the Titanic dataset and generate insights, including visualizations.

## Features
- **PlannerAgent**: Breaks down complex data analysis tasks into manageable subtasks.
- **DataAnalyserAgent**: Performs data analysis using predefined tools.
- **CodeGeneratorAgent**: Generates Python code and shell commands for data analysis.
- **CodeExecutorAgent**: Executes code in a Docker container for reproducible results.
- **UserAgent**: Simulates user feedback and approval.

## Dataset
The analysis is based on the `Titanic-Dataset.csv` file. Make sure this file is present in the project directory.

## Requirements
Install dependencies using:
```bash
pip install -r requirements.txt
```

## Environment Variables
Set your OpenAI API key in a `.env` file or as an environment variable:
```
OPENAI_API_KEY=your_openai_api_key_here
```

## Usage
Run the main script:
```bash
python be_with_humanPaid.py
```

The agents will collaborate to analyze the Titanic dataset and generate insights, including a pie chart of passengers by embarkation port.

## Notes
- The code execution is performed inside a Docker container for isolation and reproducibility.
- The script is designed for educational and demonstration purposes.

## License
MIT License 
