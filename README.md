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

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👥 Authors

- **Your Name** - *Initial work* - [YourGitHub](https://github.com/yourusername)

## 🙏 Acknowledgments

- Financial data providers
- Open-source community for machine learning libraries
- Academic advisors and research collaborators

---


For questions or support, please open an issue or contact the maintainers.
