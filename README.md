# Cellular Tumor Simulations

## Overview
This project focuses on simulating the growth of cancer cells in the human body. Using computational models and biological principles, the application provides insights into tumor dynamics under various scenarios. The simulation integrates data and methodologies from several scientific studies, including cellular automata and probabilistic modeling, to explore:

1. Growth of non-clonogenic, clonogenic, and stem-cell-based tumors.
2. The effects of apoptosis and environmental conditions on tumor progression.
3. Influence of nutrients on cancer growth. 

## Features
- Simulate different types of cancer cell growth using cellular automata and probabilistic approaches.
- Visualize tumor dynamics in real-time.
- Compare simulation results to theoretical and experimental findings from literature.
- User-friendly interface for configuring parameters and running simulations.

## Prerequisites
Ensure the following are installed on your system:
- Python (>= 3.8)
- Virtual environment tools (e.g., `venv` or `conda`)

## Setup
1. Clone the repository:
   ```bash
   git clone https://github.com/MaksymSz/Cellular-Tumor-Simulations.git
   cd Cellular-Tumor-Simulations
   ```

2. Set up a virtual environment (optional but recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage
1. Run the application:
   ```bash
   python app.py
   ```

2. Follow the on-screen instructions to configure simulation parameters and scenarios.

3. View the simulation results through the GUI or as generated plots and visualizations.

## Project Structure
- `app.py`: Main application file to run the simulation.
- `models/`: Contains the core simulation models for tumor growth.
- `visualization/`: Tools and scripts for generating plots and animations.
- `requirements.txt`: Lists the Python dependencies required to run the project.
- `README.md`: Documentation and instructions for the project.

## Authors
- Adam Stajek
- Maksym Szemer
- Ayasmaa Otgonbulag

## License
This project is licensed under [MIT License](LICENSE).

## Acknowledgments
This project was supervised by Dr. inż. Marcin Piekarczyk and conducted as part of the "Symulacja Systemów Dyskretnych" course at AGH University of Science and Technology, Kraków, Poland.

For more details, refer to the project documentation.
