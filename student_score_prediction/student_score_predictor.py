"""
Student Score Prediction Based on Study Habits
Project 2: Regression Model to predict student performance

This script demonstrates:
- Data preprocessing and cleaning
- Exploratory data analysis with visualizations
- Linear regression model training
- Model evaluation and prediction
- Dashboard-style interactive visualizations
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import warnings
warnings.filterwarnings('ignore')

# Set style for better visualizations
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class StudentScorePredictor:
    """
    A comprehensive class for predicting student scores based on study habits
    """
    
    def __init__(self, data_path):
        """Initialize the predictor with data path"""
        self.data_path = data_path
        self.data = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.model = None
        
    def load_and_clean_data(self):
        """Import and clean CSV data using pandas"""
        print("📊 Loading and cleaning data...")
        
        # Load the dataset
        self.data = pd.read_csv(self.data_path)
        
        # Display basic information
        print(f"Dataset shape: {self.data.shape}")
        print(f"Columns: {list(self.data.columns)}")
        
        # Check for missing values
        missing_values = self.data.isnull().sum()
        print(f"\nMissing values:\n{missing_values}")
        
        # Basic statistics
        print(f"\nDataset Statistics:")
        print(self.data.describe())
        
        # Check for duplicates
        duplicates = self.data.duplicated().sum()
        print(f"\nDuplicate rows: {duplicates}")
        
        # Remove duplicates if any
        if duplicates > 0:
            self.data = self.data.drop_duplicates()
            print(f"Removed {duplicates} duplicate rows")
        
        # Data validation
        self.validate_data()
        
        return self.data
    
    def validate_data(self):
        """Validate data ranges and quality"""
        print("\n🔍 Validating data quality...")
        
        # Check for valid ranges
        invalid_hours = self.data[(self.data['Hours_Studied'] < 0) | (self.data['Hours_Studied'] > 24)]
        invalid_attendance = self.data[(self.data['Attendance'] < 0) | (self.data['Attendance'] > 100)]
        invalid_scores = self.data[(self.data['Final_Score'] < 0) | (self.data['Final_Score'] > 100)]
        
        if not invalid_hours.empty:
            print(f"⚠️ Found {len(invalid_hours)} rows with invalid study hours")
        if not invalid_attendance.empty:
            print(f"⚠️ Found {len(invalid_attendance)} rows with invalid attendance")
        if not invalid_scores.empty:
            print(f"⚠️ Found {len(invalid_scores)} rows with invalid scores")
        
        if invalid_hours.empty and invalid_attendance.empty and invalid_scores.empty:
            print("✅ Data validation passed - all values are within expected ranges")
    
    def explore_data(self):
        """Perform exploratory data analysis"""
        print("\n📈 Performing Exploratory Data Analysis...")
        
        # Create subplots for comprehensive analysis
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Student Score Prediction - Exploratory Data Analysis', fontsize=16, fontweight='bold')
        
        # 1. Distribution of Final Scores
        axes[0, 0].hist(self.data['Final_Score'], bins=15, alpha=0.7, color='skyblue', edgecolor='black')
        axes[0, 0].set_title('Distribution of Final Scores')
        axes[0, 0].set_xlabel('Final Score')
        axes[0, 0].set_ylabel('Frequency')
        
        # 2. Hours Studied vs Final Score
        axes[0, 1].scatter(self.data['Hours_Studied'], self.data['Final_Score'], alpha=0.7, color='green')
        axes[0, 1].set_title('Hours Studied vs Final Score')
        axes[0, 1].set_xlabel('Hours Studied')
        axes[0, 1].set_ylabel('Final Score')
        
        # Add correlation coefficient
        corr_hours = self.data['Hours_Studied'].corr(self.data['Final_Score'])
        axes[0, 1].text(0.05, 0.95, f'Correlation: {corr_hours:.3f}', 
                       transform=axes[0, 1].transAxes, fontweight='bold',
                       bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7))
        
        # 3. Attendance vs Final Score
        axes[0, 2].scatter(self.data['Attendance'], self.data['Final_Score'], alpha=0.7, color='orange')
        axes[0, 2].set_title('Attendance vs Final Score')
        axes[0, 2].set_xlabel('Attendance (%)')
        axes[0, 2].set_ylabel('Final Score')
        
        # Add correlation coefficient
        corr_attendance = self.data['Attendance'].corr(self.data['Final_Score'])
        axes[0, 2].text(0.05, 0.95, f'Correlation: {corr_attendance:.3f}', 
                       transform=axes[0, 2].transAxes, fontweight='bold',
                       bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7))
        
        # 4. Correlation Heatmap
        correlation_matrix = self.data.corr()
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, ax=axes[1, 0])
        axes[1, 0].set_title('Correlation Matrix Heatmap')
        
        # 5. Box plots for outlier detection
        box_data = [self.data['Hours_Studied'], self.data['Attendance'], self.data['Final_Score']]
        axes[1, 1].boxplot(box_data, labels=['Hours Studied', 'Attendance', 'Final Score'])
        axes[1, 1].set_title('Box Plots for Outlier Detection')
        axes[1, 1].set_ylabel('Values')
        
        # 6. Pairplot style visualization
        axes[1, 2].scatter(self.data['Hours_Studied'], self.data['Attendance'], 
                          c=self.data['Final_Score'], cmap='viridis', alpha=0.7)
        scatter = axes[1, 2].scatter(self.data['Hours_Studied'], self.data['Attendance'], 
                                   c=self.data['Final_Score'], cmap='viridis', alpha=0.7)
        axes[1, 2].set_title('Study Hours vs Attendance (colored by Final Score)')
        axes[1, 2].set_xlabel('Hours Studied')
        axes[1, 2].set_ylabel('Attendance (%)')
        plt.colorbar(scatter, ax=axes[1, 2], label='Final Score')
        
        plt.tight_layout()
        plt.savefig('visualizations/exploratory_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Print correlation insights
        print(f"\n🔍 Key Insights:")
        print(f"• Hours Studied - Final Score correlation: {corr_hours:.3f}")
        print(f"• Attendance - Final Score correlation: {corr_attendance:.3f}")
        
        # Determine stronger predictor
        if abs(corr_hours) > abs(corr_attendance):
            print(f"• Hours Studied appears to be a stronger predictor of Final Score")
        else:
            print(f"• Attendance appears to be a stronger predictor of Final Score")
    
    def prepare_data(self, test_size=0.2, random_state=42):
        """Prepare data for training with train/test split"""
        print(f"\n🔄 Preparing data for model training...")
        
        # Define features and target
        X = self.data[['Hours_Studied', 'Attendance']]
        y = self.data['Final_Score']
        
        # Train/test split for model evaluation
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        
        print(f"Training set size: {len(self.X_train)} samples")
        print(f"Test set size: {len(self.X_test)} samples")
        print(f"Feature columns: {list(X.columns)}")
        
        return self.X_train, self.X_test, self.y_train, self.y_test
    
    def train_model(self):
        """Fit regression model using scikit-learn"""
        print(f"\n🤖 Training Linear Regression model...")
        
        # Initialize and train the model
        self.model = LinearRegression()
        self.model.fit(self.X_train, self.y_train)
        
        # Model coefficients
        print(f"Model Coefficients:")
        print(f"• Hours Studied coefficient: {self.model.coef_[0]:.3f}")
        print(f"• Attendance coefficient: {self.model.coef_[1]:.3f}")
        print(f"• Intercept: {self.model.intercept_:.3f}")
        
        # Model equation
        print(f"\nModel Equation:")
        print(f"Final_Score = {self.model.intercept_:.3f} + "
              f"{self.model.coef_[0]:.3f} * Hours_Studied + "
              f"{self.model.coef_[1]:.3f} * Attendance")
        
        return self.model
    
    def evaluate_model(self):
        """Calculate R2 score and Mean Absolute Error"""
        print(f"\n📊 Evaluating model performance...")
        
        # Make predictions
        y_train_pred = self.model.predict(self.X_train)
        y_test_pred = self.model.predict(self.X_test)
        
        # Calculate metrics for training set
        train_r2 = r2_score(self.y_train, y_train_pred)
        train_mae = mean_absolute_error(self.y_train, y_train_pred)
        train_rmse = np.sqrt(mean_squared_error(self.y_train, y_train_pred))
        
        # Calculate metrics for test set
        test_r2 = r2_score(self.y_test, y_test_pred)
        test_mae = mean_absolute_error(self.y_test, y_test_pred)
        test_rmse = np.sqrt(mean_squared_error(self.y_test, y_test_pred))
        
        print(f"Training Set Performance:")
        print(f"• R² Score: {train_r2:.4f}")
        print(f"• Mean Absolute Error: {train_mae:.3f}")
        print(f"• Root Mean Square Error: {train_rmse:.3f}")
        
        print(f"\nTest Set Performance:")
        print(f"• R² Score: {test_r2:.4f}")
        print(f"• Mean Absolute Error: {test_mae:.3f}")
        print(f"• Root Mean Square Error: {test_rmse:.3f}")
        
        # Model interpretation
        if test_r2 > 0.8:
            print(f"\n✅ Excellent model performance! The model explains {test_r2*100:.1f}% of the variance.")
        elif test_r2 > 0.6:
            print(f"\n👍 Good model performance! The model explains {test_r2*100:.1f}% of the variance.")
        else:
            print(f"\n⚠️ Model performance could be improved. The model explains {test_r2*100:.1f}% of the variance.")
        
        # Check for overfitting
        if train_r2 - test_r2 > 0.1:
            print(f"⚠️ Potential overfitting detected (training R² - test R² = {train_r2 - test_r2:.3f})")
        else:
            print(f"✅ No significant overfitting detected")
        
        return {
            'train_r2': train_r2, 'test_r2': test_r2,
            'train_mae': train_mae, 'test_mae': test_mae,
            'train_rmse': train_rmse, 'test_rmse': test_rmse
        }
    
    def visualize_predictions(self):
        """Create comprehensive visualization of model predictions"""
        print(f"\n📈 Creating prediction visualizations...")
        
        # Make predictions
        y_test_pred = self.model.predict(self.X_test)
        
        # Create visualization
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Model Prediction Analysis', fontsize=16, fontweight='bold')
        
        # 1. Actual vs Predicted scatter plot
        axes[0, 0].scatter(self.y_test, y_test_pred, alpha=0.7, color='blue')
        axes[0, 0].plot([self.y_test.min(), self.y_test.max()], 
                       [self.y_test.min(), self.y_test.max()], 'r--', lw=2)
        axes[0, 0].set_xlabel('Actual Final Score')
        axes[0, 0].set_ylabel('Predicted Final Score')
        axes[0, 0].set_title('Actual vs Predicted Scores')
        
        # Add R² score to the plot
        r2 = r2_score(self.y_test, y_test_pred)
        axes[0, 0].text(0.05, 0.95, f'R² = {r2:.4f}', transform=axes[0, 0].transAxes,
                       fontweight='bold', bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7))
        
        # 2. Residuals plot
        residuals = self.y_test - y_test_pred
        axes[0, 1].scatter(y_test_pred, residuals, alpha=0.7, color='green')
        axes[0, 1].axhline(y=0, color='red', linestyle='--')
        axes[0, 1].set_xlabel('Predicted Final Score')
        axes[0, 1].set_ylabel('Residuals')
        axes[0, 1].set_title('Residuals Plot')
        
        # 3. Feature importance visualization
        feature_names = ['Hours Studied', 'Attendance']
        coefficients = self.model.coef_
        axes[1, 0].bar(feature_names, coefficients, color=['skyblue', 'lightcoral'])
        axes[1, 0].set_title('Feature Coefficients')
        axes[1, 0].set_ylabel('Coefficient Value')
        
        # Add coefficient values on bars
        for i, v in enumerate(coefficients):
            axes[1, 0].text(i, v + 0.1 if v > 0 else v - 0.3, f'{v:.3f}', 
                           ha='center', fontweight='bold')
        
        # 4. Prediction error distribution
        axes[1, 1].hist(residuals, bins=10, alpha=0.7, color='purple', edgecolor='black')
        axes[1, 1].axvline(x=0, color='red', linestyle='--')
        axes[1, 1].set_xlabel('Prediction Error')
        axes[1, 1].set_ylabel('Frequency')
        axes[1, 1].set_title('Distribution of Prediction Errors')
        
        plt.tight_layout()
        plt.savefig('visualizations/prediction_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def predict_score(self, hours_studied, attendance):
        """Predict new student scores based on input"""
        if self.model is None:
            raise ValueError("Model has not been trained yet. Please train the model first.")
        
        # Create input array
        input_data = np.array([[hours_studied, attendance]])
        
        # Make prediction
        predicted_score = self.model.predict(input_data)[0]
        
        # Calculate confidence interval (simplified)
        # In practice, you might want to use prediction intervals
        y_test_pred = self.model.predict(self.X_test)
        residuals = self.y_test - y_test_pred
        std_residual = np.std(residuals)
        
        confidence_interval = 1.96 * std_residual  # 95% confidence interval
        
        return {
            'predicted_score': predicted_score,
            'confidence_lower': predicted_score - confidence_interval,
            'confidence_upper': predicted_score + confidence_interval
        }
    
    def create_dashboard_visualization(self):
        """Create an interactive-style dashboard visualization"""
        print(f"\n📊 Creating dashboard visualization...")
        
        # Create a comprehensive dashboard
        fig = plt.figure(figsize=(20, 14))
        
        # Main title
        fig.suptitle('Student Score Prediction Dashboard', fontsize=20, fontweight='bold', y=0.95)
        
        # Create a grid layout
        gs = fig.add_gridspec(3, 4, hspace=0.3, wspace=0.3)
        
        # 1. Dataset Overview (top-left)
        ax1 = fig.add_subplot(gs[0, 0])
        dataset_info = [
            f"Total Students: {len(self.data)}",
            f"Avg Hours Studied: {self.data['Hours_Studied'].mean():.1f}",
            f"Avg Attendance: {self.data['Attendance'].mean():.1f}%",
            f"Avg Final Score: {self.data['Final_Score'].mean():.1f}"
        ]
        ax1.text(0.1, 0.7, "\n".join(dataset_info), fontsize=12, fontweight='bold',
                transform=ax1.transAxes, verticalalignment='top')
        ax1.set_title('Dataset Overview', fontweight='bold')
        ax1.axis('off')
        
        # 2. Model Performance Metrics (top-center-left)
        ax2 = fig.add_subplot(gs[0, 1])
        y_test_pred = self.model.predict(self.X_test)
        r2 = r2_score(self.y_test, y_test_pred)
        mae = mean_absolute_error(self.y_test, y_test_pred)
        
        metrics_info = [
            f"R² Score: {r2:.4f}",
            f"MAE: {mae:.3f}",
            f"Accuracy: {r2*100:.1f}%"
        ]
        ax2.text(0.1, 0.7, "\n".join(metrics_info), fontsize=12, fontweight='bold',
                transform=ax2.transAxes, verticalalignment='top')
        ax2.set_title('Model Performance', fontweight='bold')
        ax2.axis('off')
        
        # 3. Hours vs Score (top-center-right)
        ax3 = fig.add_subplot(gs[0, 2])
        ax3.scatter(self.data['Hours_Studied'], self.data['Final_Score'], alpha=0.6, color='blue')
        ax3.set_xlabel('Hours Studied')
        ax3.set_ylabel('Final Score')
        ax3.set_title('Study Hours Impact')
        
        # 4. Attendance vs Score (top-right)
        ax4 = fig.add_subplot(gs[0, 3])
        ax4.scatter(self.data['Attendance'], self.data['Final_Score'], alpha=0.6, color='green')
        ax4.set_xlabel('Attendance (%)')
        ax4.set_ylabel('Final Score')
        ax4.set_title('Attendance Impact')
        
        # 5. Score Distribution (middle-left)
        ax5 = fig.add_subplot(gs[1, 0])
        ax5.hist(self.data['Final_Score'], bins=15, alpha=0.7, color='skyblue', edgecolor='black')
        ax5.set_xlabel('Final Score')
        ax5.set_ylabel('Frequency')
        ax5.set_title('Score Distribution')
        
        # 6. Actual vs Predicted (middle-center-left)
        ax6 = fig.add_subplot(gs[1, 1])
        ax6.scatter(self.y_test, y_test_pred, alpha=0.7, color='red')
        ax6.plot([self.y_test.min(), self.y_test.max()], 
                [self.y_test.min(), self.y_test.max()], 'k--', lw=2)
        ax6.set_xlabel('Actual Score')
        ax6.set_ylabel('Predicted Score')
        ax6.set_title('Prediction Accuracy')
        
        # 7. Feature Importance (middle-center-right)
        ax7 = fig.add_subplot(gs[1, 2])
        features = ['Hours Studied', 'Attendance']
        importance = np.abs(self.model.coef_)
        bars = ax7.bar(features, importance, color=['lightblue', 'lightcoral'])
        ax7.set_title('Feature Importance')
        ax7.set_ylabel('Coefficient Magnitude')
        
        # Add values on bars
        for bar, val in zip(bars, importance):
            ax7.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                    f'{val:.3f}', ha='center', fontweight='bold')
        
        # 8. Residuals (middle-right)
        ax8 = fig.add_subplot(gs[1, 3])
        residuals = self.y_test - y_test_pred
        ax8.scatter(y_test_pred, residuals, alpha=0.7, color='orange')
        ax8.axhline(y=0, color='red', linestyle='--')
        ax8.set_xlabel('Predicted Score')
        ax8.set_ylabel('Residuals')
        ax8.set_title('Prediction Residuals')
        
        # 9. Correlation Heatmap (bottom span)
        ax9 = fig.add_subplot(gs[2, :2])
        corr_matrix = self.data.corr()
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, ax=ax9,
                   square=True, cbar_kws={'shrink': 0.8})
        ax9.set_title('Correlation Matrix')
        
        # 10. Prediction Example (bottom-right span)
        ax10 = fig.add_subplot(gs[2, 2:])
        
        # Example prediction
        example_hours = 4
        example_attendance = 80
        prediction_result = self.predict_score(example_hours, example_attendance)
        
        prediction_text = [
            "PREDICTION EXAMPLE:",
            f"Study Hours: {example_hours}",
            f"Attendance: {example_attendance}%",
            "",
            f"Predicted Score: {prediction_result['predicted_score']:.1f}",
            f"95% Confidence Interval:",
            f"({prediction_result['confidence_lower']:.1f} - {prediction_result['confidence_upper']:.1f})"
        ]
        
        ax10.text(0.1, 0.8, "\n".join(prediction_text), fontsize=14, fontweight='bold',
                 transform=ax10.transAxes, verticalalignment='top',
                 bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgreen", alpha=0.7))
        ax10.set_title('Sample Prediction', fontweight='bold')
        ax10.axis('off')
        
        plt.savefig('visualizations/dashboard.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def run_complete_analysis(self):
        """Run the complete analysis pipeline"""
        print("🚀 Starting Student Score Prediction Analysis")
        print("=" * 50)
        
        # Load and clean data
        self.load_and_clean_data()
        
        # Explore data
        self.explore_data()
        
        # Prepare data
        self.prepare_data()
        
        # Train model
        self.train_model()
        
        # Evaluate model
        metrics = self.evaluate_model()
        
        # Visualize predictions
        self.visualize_predictions()
        
        # Create dashboard
        self.create_dashboard_visualization()
        
        # Example prediction as requested
        print("\n" + "="*50)
        print("🎯 EXPECTED OUTPUT DEMONSTRATION")
        print("="*50)
        
        example_prediction = self.predict_score(4, 80)
        print(f"\nPrediction for student with 4 study hours and 80% attendance:")
        print(f"• Predicted Final Score: {example_prediction['predicted_score']:.1f}")
        print(f"• 95% Confidence Interval: ({example_prediction['confidence_lower']:.1f} - {example_prediction['confidence_upper']:.1f})")
        
        print(f"\nModel Error Metrics:")
        print(f"• R² Score: {metrics['test_r2']:.4f} (explains {metrics['test_r2']*100:.1f}% of variance)")
        print(f"• Mean Absolute Error: {metrics['test_mae']:.3f} points")
        print(f"• Root Mean Square Error: {metrics['test_rmse']:.3f} points")
        
        print(f"\n✅ Analysis completed successfully!")
        print(f"📁 Visualizations saved in 'visualizations/' folder")
        
        return metrics

def main():
    """Main function to run the analysis"""
    print("🎓 Student Score Prediction Project")
    print("Project 2: Predicting student performance using Machine Learning")
    print("="*60)
    
    # Initialize the predictor
    predictor = StudentScorePredictor('data/student_data.csv')
    
    # Run complete analysis
    results = predictor.run_complete_analysis()
    
    # Additional ethical considerations
    print("\n" + "="*50)
    print("🛡️ ETHICAL CONSIDERATIONS & BIAS AWARENESS")
    print("="*50)
    
    ethical_notes = [
        "• This model should be used as a supportive tool, not for punitive measures",
        "• Consider potential biases in data collection (socioeconomic factors, learning disabilities)",
        "• Model predictions should be combined with qualitative assessments",
        "• Regular model updates needed to account for changing educational environments",
        "• Ensure student privacy and data protection compliance",
        "• Consider fairness across different demographic groups"
    ]
    
    for note in ethical_notes:
        print(note)
    
    return predictor, results

if __name__ == "__main__":
    predictor, results = main()
