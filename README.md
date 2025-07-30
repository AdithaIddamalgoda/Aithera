# Uncertainty-Aware Remaining Useful Life (RUL) Prediction

This implementation provides a comprehensive solution for uncertainty-aware RUL prediction using the NASA C-MAPSS dataset. The methodology combines Monte Carlo Dropout, Deep Ensembles, calibration with temperature scaling, and an abstention mechanism for high-uncertainty predictions.

## Features

- **Monte Carlo Dropout**: Estimates epistemic uncertainty through stochastic forward passes
- **Deep Ensembles**: Uses multiple models with different initializations for robust predictions
- **Temperature Scaling**: Calibrates prediction confidence to align with empirical correctness
- **Abstention Mechanism**: Flags high-uncertainty predictions for human review
- **Comprehensive Evaluation**: Provides RMSE, calibration metrics, and uncertainty quantification

## Requirements

Install the required dependencies:

```bash
pip install -r requirements.txt
```

Required packages:
- TensorFlow 2.8+
- NumPy 1.21+
- Pandas 1.3+
- Scikit-learn 1.0+
- Matplotlib 3.5+
- Seaborn 0.11+

## Dataset

The implementation uses the NASA C-MAPSS (Commercial Modular Aero-Propulsion System Simulation) dataset. The dataset should be placed in the following structure:

```
Code/
├── data/
│   └── raw/
│       └── CMaps/
│           ├── train_FD001.txt
│           ├── test_FD001.txt
│           ├── RUL_FD001.txt
│           └── readme.txt
├── uncertainty_aware_rul_prediction.py
├── test_implementation.py
└── requirements.txt
```

## Usage

### Quick Test

To verify the implementation works correctly with a small subset of data:

```bash
python test_implementation.py
```

This runs a simplified version with:
- Smaller sequence length (10 instead of 30)
- Fewer ensemble models (2 instead of 5)
- Fewer MC samples (10 instead of 50)
- Limited training epochs (5 instead of 50)

### Full Training

To run the complete training with the full methodology:

```bash
python uncertainty_aware_rul_prediction.py
```

This will:
1. Load and preprocess the NASA C-MAPSS FD001 dataset
2. Train an ensemble of 5 neural networks
3. Perform temperature scaling calibration
4. Evaluate on test data with uncertainty quantification
5. Generate comprehensive visualizations
6. Save trained models and results

## Methodology Details

### Neural Network Architecture

- **LSTM Layers**: Two LSTM layers (64 and 32 units) for temporal modeling
- **Dense Layers**: Three fully connected layers (128, 64, 32 units) with ReLU activation
- **Dropout Layers**: Applied after each layer (rate: 0.3) for Monte Carlo Dropout
- **Output Layer**: Single linear unit for RUL regression

### Data Preprocessing

1. **Feature Selection**: Uses operational settings and sensor measurements (excludes constant sensors)
2. **Normalization**: MinMaxScaler for both features and targets
3. **Sequence Creation**: Sliding window approach (default: 30 cycles)
4. **Train/Val/Test Split**: 80%/10%/10% split

### Uncertainty Quantification

- **Epistemic Uncertainty**: Variance between ensemble model predictions
- **Aleatoric Uncertainty**: Average within-model variance from MC Dropout
- **Total Uncertainty**: Sum of epistemic and aleatoric uncertainties

### Calibration

Temperature scaling optimizes prediction confidence using Expected Calibration Error (ECE) on validation data.

### Abstention

Predictions are flagged for human review if the 95% confidence interval width exceeds a threshold (default: 50 cycles).

## Output

### Saved Models
- `saved_models/`: Directory containing trained ensemble models, scalers, and configuration

### Visualizations
- `visualizations/prediction_analysis.png`: Predicted vs actual RUL, uncertainty intervals
- `visualizations/calibration_plot.png`: Reliability diagram and confidence distribution

### Metrics
- **RMSE**: Root Mean Square Error for prediction accuracy
- **Coverage**: Percentage of true values within confidence intervals
- **Abstention Rate**: Percentage of predictions flagged for review
- **ECE**: Expected Calibration Error

## Configuration

Key parameters can be modified in the `RULPredictor` class initialization:

```python
predictor = RULPredictor(
    sequence_length=30,    # Input sequence length
    n_features=24,         # Number of features (auto-detected)
    n_models=5,           # Number of ensemble models
    mc_samples=50         # Monte Carlo samples for uncertainty
)
```

## Example Results

Typical performance metrics:
- RMSE: 15-25 cycles
- 95% Coverage: 90-95%
- Abstention Rate: 10-20%
- Training Time: 30-60 minutes (depending on hardware)

## Troubleshooting

### Common Issues

1. **File Not Found**: Ensure dataset files are in the correct directory structure
2. **Memory Error**: Reduce batch size or sequence length for limited memory
3. **Slow Training**: Reduce number of models or MC samples for faster execution

### Performance Tips

- Use GPU acceleration if available (TensorFlow will auto-detect CUDA)
- Adjust batch size based on available memory
- Consider reducing ensemble size for faster prototyping

## References

- Saxena, A., et al. "Damage Propagation Modeling for Aircraft Engine Run-to-Failure Simulation" (2008)
- Gal, Y., & Ghahramani, Z. "Dropout as a Bayesian Approximation" (2016)
- Lakshminarayanan, B., et al. "Simple and Scalable Predictive Uncertainty Estimation" (2017)

## License

This implementation is provided for research and educational purposes.
