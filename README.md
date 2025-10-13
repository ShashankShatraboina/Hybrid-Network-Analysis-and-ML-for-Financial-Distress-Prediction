# Hybrid Network Analysis and Machine Learning for Financial Distress Prediction

![Python](https://img.shields.io/badge/Python-3.7%2B-blue)
![Machine Learning](https://img.shields.io/badge/Machine-Learning-orange)
![Network Analysis](https://img.shields.io/badge/Network-Analysis-green)
![Finance](https://img.shields.io/badge/Finance-Prediction-lightgrey)

A novel hybrid approach combining network analysis and machine learning techniques for enhanced financial distress prediction in corporate entities.

## 📖 Abstract

Financial distress prediction is crucial for financial planning, particularly amid emerging uncertainties. This study introduces an innovative methodology that amalgamates network analysis and machine learning techniques to predict financial distress. The approach involves establishing company networks based on similarity and correlation in crucial financial indicators, extracting network-centric features, and integrating them with traditional financial variables to significantly improve predictive accuracy.

## 🚀 Key Features

- **Dual Network Construction**: Creates similarity-based and correlation-based company networks
- **Network Feature Extraction**: Derives 7 key network metrics (Degree Centrality, Betweenness, Closeness, etc.)
- **Community Detection**: Applies label propagation for company clustering
- **Multi-Model Evaluation**: Tests 5 classification algorithms across different scenarios
- **Enhanced Accuracy**: Demonstrates superior predictive capabilities over traditional methods

## 🏗️ System Architecture

### Core Modules

1. **Data Preprocessing & Feature Selection**
   - Handling missing and duplicate data
   - Feature correlation analysis
   - Selection of most influential financial indicators

2. **Network Construction**
   - **Similarity Network**: Based on distances across 5 key financial features
   - **Correlation Network**: Based on correlation coefficients of critical features
   - K-Nearest Neighbors algorithm for network formation

3. **Network Analysis**
   - Feature Extraction (7 network metrics)
   - Community Detection using label propagation
   - Integration of network features into dataset

4. **Machine Learning Models**
   - Logistic Regression
   - Decision Tree Classifier
   - K-Nearest Neighbors
   - Support Vector Machine
   - Passive Aggressive Classifier

## 📊 Methodology

### Network Features Extracted

- Degree Centrality
- Betweenness Centrality
- Closeness Centrality
- Clustering Coefficient
- Page Rank Centrality
- Average Neighbor Degree
- Clustering Coefficient Weighted

### Evaluation Metrics

- Accuracy
- Precision
- Recall
- F1-Score

## 🛠️ Installation

### Prerequisites

- Python 3.7+
- Required libraries (see requirements.txt)

### Setup

```bash
# Clone the repository
git clone https://github.com/yourusername/financial-distress-prediction.git

# Navigate to project directory
cd financial-distress-prediction

# Install dependencies
pip install -r requirements.txt
```

### Required Libraries

```txt
pandas>=1.3.0
numpy>=1.21.0
scikit-learn>=1.0.0
networkx>=2.6.0
matplotlib>=3.4.0
seaborn>=0.11.0
django>=4.0.0
```

## 💻 Usage

### Data Preparation

1. Prepare your financial dataset in CSV format
2. Ensure required financial indicators are included
3. Preprocess data using the provided scripts

### Running the Application

```bash
# Start the Django server
python manage.py runserver

# Access the web interface at http://localhost:8000
```

### Model Training

```python
from src.models.hybrid_predictor import HybridPredictor

# Initialize the predictor
predictor = HybridPredictor()

# Load and preprocess data
predictor.load_data('path/to/financial_data.csv')

# Build networks and extract features
predictor.build_networks()

# Train models
predictor.train_models()

# Make predictions
predictions = predictor.predict(new_data)
```

## 📈 Results

The hybrid model demonstrates:

- **Significant improvement** in prediction accuracy compared to traditional methods
- **Enhanced capability** to capture complex relationships between companies
- **Robust performance** across different market conditions
- **Superior early warning** capabilities for financial distress

## 🎯 Key Findings

1. **Network Features Matter**: Features from similarity networks play a pivotal role in improving predictive accuracy
2. **Hybrid Approach Superior**: Combination of network analysis and ML outperforms standalone methods
3. **Real-time Adaptability**: System can adapt to changing market conditions
4. **Comprehensive Insight**: Provides holistic understanding of financial entity interactions

## 📁 Project Structure

```
financial-distress-prediction/
│
├── data/                    # Dataset files
├── src/
│   ├── preprocessing/       # Data preprocessing modules
│   ├── network_analysis/    # Network construction and analysis
│   ├── models/             # Machine learning models
│   └── visualization/      # Result visualization
├── web_app/                # Django web application
├── notebooks/              # Jupyter notebooks for analysis
├── tests/                  # Test cases
└── docs/                   # Documentation
```

## 🔬 Research Contribution

This work contributes to financial analytics by:

- Introducing a novel network-based approach for financial distress prediction
- Demonstrating the value of inter-company relationships in risk assessment
- Providing a framework for integrating network science with financial modeling
- Offering practical tools for investors and financial analysts

## 📚 Citation

If you use this code in your research, please cite:

```bibtex
@article{hybridfinancial2024,
  title={A Hybrid Network Analysis and Machine Learning Model for Enhanced Financial Distress Prediction},
  author={Your Name},
  journal={GitHub Repository},
  year={2024}
}
```

## 🤝 Contributing

We welcome contributions! Please feel free to submit pull requests, report bugs, or suggest new features.

## 👥 Authors

- **Shashank Shatraboina** 

## 🙏 Acknowledgments

- Financial data providers
- Open-source community for machine learning libraries
- Academic advisors and research collaborators

---





<h2 align="center">
Movix - Ultimate Movie and TV Show Discovery Platform</h2>

<p align="center"><img src="./src/assets/screenshort/1.PNG" alt="Movix homepage"></p>

<p>Movix is a web application built using React and Redux that allows users to search for movies and TV shows, view popular, trending, and upcoming releases on a daily and weekly basis, and explore detailed information about each title, including trailers and related videos.</p>

<h3>📝 Features</h3>

- <strong>Movie and TV Shows Search</strong>: Users can easily search for movies and TV shows by their respective names.

- <strong>Popular, Trending, and Upcoming:</strong> The homepage showcases popular, trending, and upcoming movies and TV shows, with their names, posters, genres, and ratings.

- <strong>Detailed Movie/TV Show Pages</strong>: Clicking on a movie or TV show provides users with a detailed page containing comprehensive information about the title, cast, runtime, release year, rating, director, writer, including its description, trailer, and additional videos.

- <strong>Personalized Recommendations</strong>: Users receive recommendations for similar movies and TV shows based on the content they are currently viewing.

- <strong>Fine-Tuned Filters</strong>: Users can apply filters based on various criteria, such as genre, release date, rating, and more, to refine their search results.

<h3>🚀 Live Demo</h3>

[https://movix-taupe.vercel.app](https://movix-taupe.vercel.app)

<h5>Tending & Popular Movies:</h5>

<img src="./src/assets/screenshort/4.PNG" alt="Movix homepage">
<img src="./src/assets/screenshort/5.PNG" alt="Movix homepage">

<h5>Movie Details Page:</h5>

<img src="./src/assets/screenshort/7.PNG" alt="Movix homepage">
<img src="./src/assets/screenshort/8.PNG" alt="Movix homepage">

<h5>Search Results Page:</h5>

<img src="./src/assets/screenshort/13.PNG" alt="Movix homepage">

<h5>Explore Movies & TV Shows:</h5>

<img src="./src/assets/screenshort/11.PNG" alt="Movix homepage">
<img src="./src/assets/screenshort/12.PNG" alt="Movix homepage">

<h3>🛠️ Installation Steps:</h3>

<p>1. Clone the repository</p>

```
git clone https://github.com/masud-rana44/Movix.git
```

<p>2. Install the required dependencies </p>

```
npm install
```

<p>3. Start the development server</p>

```
npm run dev
```

<p>4. Access the application at</p>

```
http://localhost:5173
```

<h3>💻 Built with</h3>

Technologies used in the project:

- [React](#) - Building user interfaces
- [Redux](#) - UI state management
- [scss](#) - For styling
- [Axios](#) - API requests to the TMDB API
- [React Router](#) - Navigation and routing within the application

<h3>🙏 Acknowledgments</h3>

In the development of Movix, we express our gratitude to the following third-party libraries and APIs that have significantly contributed to the application's functionality and user experience:

- **Redux Toolkit (`@reduxjs/toolkit`):** A comprehensive toolset for managing application state with Redux, streamlining state management and reducing boilerplate code.

- **Axios (`axios`):** A reliable and efficient HTTP client that seamlessly integrates with the TMDB API, enabling smooth data retrieval.

- **Day.js (`dayjs`):** A lightweight and versatile library for date and time manipulation, enhancing the application's date formatting capabilities.

- **React (`react`) and React DOM (`react-dom`):** The core libraries powering the dynamic user interface and rendering of React components.

- **React Circular Progressbar (`react-circular-progressbar`):** An eye-catching component that brings visually appealing circular progress bars to the application.

- **React Icons (`react-icons`):** A treasure trove of icons that adds visual charm and enhances the user interface with diverse iconography.

- **React Infinite Scroll Component (`react-infinite-scroll-component`):** Empowers infinite scrolling functionality, making content loading seamless and intuitive.

- **React Lazy Load Image Component (`react-lazy-load-image-component`):** Enhances performance by deferring image loading until needed, improving page loading times.

- **React Player (`react-player`):** Facilitates smooth integration of media players to showcase movie trailers and videos within the application.

- **React Redux (`react-redux`):** Seamlessly integrates Redux state management with React, providing predictable application state handling.

- **React Router DOM (`react-router-dom`):** Enables smooth and intuitive navigation and routing within the application.

- **React Select (`react-select`):** Provides customizable select dropdowns for better user interaction and search functionalities.

- **Sass (`sass`):** A powerful CSS preprocessor that streamlines and organizes styling, contributing to the overall visual aesthetics of the application.

<h3>⚠️ Disclaimer</h3>

Please note that Movix relies on the TMDB API to fetch movie and TV show data. The accuracy, completeness, and availability of the data are subject to TMDB's policies and may be subject to change. Users are advised to refer to the TMDB API documentation and terms of use for any restrictions or usage guidelines related to the data accessed through the API.

<h3>💖Like my work?</h3>

This project needs a ⭐️ from you. Don't forget to leave a star ⭐️.
