import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import os

# Set page configuration
st.set_page_config(
    page_title="Student Score Prediction Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #1f77b4;
    }
    .prediction-box {
        background-color: #e8f4fd;
        padding: 2rem;
        border-radius: 15px;
        border: 2px solid #1f77b4;
        text-align: center;
    }
    .sidebar-info {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 10px;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# Title and description
st.markdown('<h1 class="main-header">📚 Student Score Prediction Dashboard</h1>', unsafe_allow_html=True)
st.markdown("### Predict student performance based on study hours and attendance using Machine Learning")

# Load or create data
@st.cache_data
def load_data():
    data_path = "data/student_data.csv"
    if os.path.exists(data_path):
        return pd.read_csv(data_path)
    else:
        # Create sample data if file doesn't exist
        np.random.seed(42)
        n_samples = 200
        
        hours_studied = np.random.uniform(1, 10, n_samples)
        attendance = np.random.uniform(40, 100, n_samples)
        
        # Create realistic final scores with some noise
        final_score = (hours_studied * 8 + attendance * 0.6 + np.random.normal(0, 5, n_samples))
        final_score = np.clip(final_score, 0, 100)
        
        data = pd.DataFrame({
            'Hours_Studied': hours_studied,
            'Attendance': attendance,
            'Final_Score': final_score
        })
        
        # Create directory if it doesn't exist
        os.makedirs("data", exist_ok=True)
        data.to_csv(data_path, index=False)
        return data

# Load and train model
@st.cache_data
def train_model(data):
    X = data[['Hours_Studied', 'Attendance']]
    y = data['Final_Score']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    
    metrics = {
        'r2_score': r2_score(y_test, y_pred),
        'mse': mean_squared_error(y_test, y_pred),
        'mae': mean_absolute_error(y_test, y_pred),
        'rmse': np.sqrt(mean_squared_error(y_test, y_pred))
    }
    
    return model, metrics, X_test, y_test, y_pred

# Load data and train model
data = load_data()
model, metrics, X_test, y_test, y_pred = train_model(data)

# Sidebar for user inputs and controls
st.sidebar.markdown('<div class="sidebar-info"><h3>🎯 Make Predictions</h3></div>', unsafe_allow_html=True)

# Prediction inputs
st.sidebar.markdown("#### Enter Student Details:")
hours_input = st.sidebar.slider(
    "📖 Hours Studied per Week", 
    min_value=1.0, 
    max_value=15.0, 
    value=7.0, 
    step=0.5,
    help="Number of hours the student studies per week"
)

attendance_input = st.sidebar.slider(
    "📅 Attendance Percentage", 
    min_value=0.0, 
    max_value=100.0, 
    value=85.0, 
    step=1.0,
    help="Student's attendance percentage"
)

# Make prediction
if st.sidebar.button("🔮 Predict Score", type="primary"):
    prediction = model.predict([[hours_input, attendance_input]])[0]
    st.sidebar.markdown(f"""
    <div class="prediction-box">
        <h3>Predicted Final Score</h3>
        <h1 style="color: #1f77b4;">{prediction:.1f}/100</h1>
    </div>
    """, unsafe_allow_html=True)

# Sidebar model information
st.sidebar.markdown('<div class="sidebar-info"><h3>📊 Model Performance</h3></div>', unsafe_allow_html=True)
st.sidebar.metric("R² Score", f"{metrics['r2_score']:.3f}")
st.sidebar.metric("Mean Absolute Error", f"{metrics['mae']:.2f}")
st.sidebar.metric("Root Mean Square Error", f"{metrics['rmse']:.2f}")

# Navigation tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Data Overview", 
    "📈 Exploratory Analysis", 
    "🤖 Model Performance", 
    "🔍 Interactive Predictions",
    "📋 Data Insights"
])

with tab1:
    st.header("📊 Dataset Overview")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Students", len(data))
    with col2:
        st.metric("Average Hours Studied", f"{data['Hours_Studied'].mean():.1f}")
    with col3:
        st.metric("Average Attendance", f"{data['Attendance'].mean():.1f}%")
    with col4:
        st.metric("Average Final Score", f"{data['Final_Score'].mean():.1f}")
    
    st.subheader("📋 Raw Data Sample")
    st.dataframe(data.head(10), use_container_width=True)
    
    st.subheader("📈 Statistical Summary")
    st.dataframe(data.describe(), use_container_width=True)

with tab2:
    st.header("📈 Exploratory Data Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Correlation heatmap
        st.subheader("🔥 Correlation Matrix")
        fig_corr = px.imshow(
            data.corr(),
            text_auto=True,
            aspect="auto",
            color_continuous_scale="RdBu",
            title="Feature Correlation Matrix"
        )
        fig_corr.update_layout(height=400)
        st.plotly_chart(fig_corr, use_container_width=True)
    
    with col2:
        # Distribution plots
        st.subheader("📊 Score Distribution")
        fig_hist = px.histogram(
            data, 
            x="Final_Score", 
            nbins=20,
            title="Distribution of Final Scores",
            color_discrete_sequence=["#1f77b4"]
        )
        fig_hist.update_layout(height=400)
        st.plotly_chart(fig_hist, use_container_width=True)
    
    # Scatter plots
    col3, col4 = st.columns(2)
    
    with col3:
        st.subheader("📖 Hours vs Score")
        fig_scatter1 = px.scatter(
            data, 
            x="Hours_Studied", 
            y="Final_Score",
            trendline="ols",
            title="Study Hours vs Final Score",
            color="Final_Score",
            color_continuous_scale="viridis"
        )
        st.plotly_chart(fig_scatter1, use_container_width=True)
    
    with col4:
        st.subheader("📅 Attendance vs Score")
        fig_scatter2 = px.scatter(
            data, 
            x="Attendance", 
            y="Final_Score",
            trendline="ols",
            title="Attendance vs Final Score",
            color="Final_Score",
            color_continuous_scale="viridis"
        )
        st.plotly_chart(fig_scatter2, use_container_width=True)
    
    # 3D Scatter plot
    st.subheader("🌐 3D Relationship Visualization")
    fig_3d = px.scatter_3d(
        data, 
        x="Hours_Studied", 
        y="Attendance", 
        z="Final_Score",
        color="Final_Score",
        size_max=10,
        title="3D View: Hours Studied vs Attendance vs Final Score",
        color_continuous_scale="viridis"
    )
    fig_3d.update_layout(height=600)
    st.plotly_chart(fig_3d, use_container_width=True)

with tab3:
    st.header("🤖 Model Performance Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Performance metrics
        st.subheader("📊 Model Metrics")
        
        metrics_df = pd.DataFrame({
            'Metric': ['R² Score', 'Mean Absolute Error', 'Root Mean Square Error', 'Mean Square Error'],
            'Value': [
                f"{metrics['r2_score']:.4f}",
                f"{metrics['mae']:.2f}",
                f"{metrics['rmse']:.2f}",
                f"{metrics['mse']:.2f}"
            ],
            'Interpretation': [
                'Higher is better (max 1.0)',
                'Lower is better',
                'Lower is better',
                'Lower is better'
            ]
        })
        st.dataframe(metrics_df, use_container_width=True, hide_index=True)
        
        # Feature importance (coefficients)
        st.subheader("🎯 Feature Importance")
        coefficients = model.coef_
        features = ['Hours_Studied', 'Attendance']
        
        coef_df = pd.DataFrame({
            'Feature': features,
            'Coefficient': coefficients,
            'Importance': np.abs(coefficients)
        }).sort_values('Importance', ascending=False)
        
        fig_coef = px.bar(
            coef_df, 
            x='Feature', 
            y='Coefficient',
            title="Feature Coefficients (Impact on Final Score)",
            color='Coefficient',
            color_continuous_scale="RdBu"
        )
        st.plotly_chart(fig_coef, use_container_width=True)
    
    with col2:
        # Actual vs Predicted
        st.subheader("🎯 Actual vs Predicted")
        
        test_results = pd.DataFrame({
            'Actual': y_test,
            'Predicted': y_pred
        })
        
        fig_actual_pred = px.scatter(
            test_results, 
            x="Actual", 
            y="Predicted",
            title="Actual vs Predicted Scores",
            trendline="ols"
        )
        fig_actual_pred.add_shape(
            type="line",
            x0=test_results['Actual'].min(),
            y0=test_results['Actual'].min(),
            x1=test_results['Actual'].max(),
            y1=test_results['Actual'].max(),
            line=dict(color="red", dash="dash")
        )
        st.plotly_chart(fig_actual_pred, use_container_width=True)
        
        # Residuals plot
        st.subheader("📉 Residuals Analysis")
        residuals = y_test - y_pred
        
        fig_residuals = px.scatter(
            x=y_pred, 
            y=residuals,
            title="Residuals vs Predicted Values",
            labels={'x': 'Predicted Values', 'y': 'Residuals'}
        )
        fig_residuals.add_hline(y=0, line_dash="dash", line_color="red")
        st.plotly_chart(fig_residuals, use_container_width=True)

with tab4:
    st.header("🔍 Interactive Prediction Tool")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("🎛️ Prediction Controls")
        
        # Advanced input controls
        hours_range = st.slider(
            "Hours Studied Range", 
            min_value=1.0, 
            max_value=15.0, 
            value=(3.0, 8.0),
            step=0.5
        )
        
        attendance_range = st.slider(
            "Attendance Range", 
            min_value=0.0, 
            max_value=100.0, 
            value=(60.0, 95.0),
            step=5.0
        )
        
        # Batch prediction
        st.subheader("📊 Batch Predictions")
        if st.button("Generate Prediction Grid"):
            hours_grid = np.linspace(hours_range[0], hours_range[1], 10)
            attendance_grid = np.linspace(attendance_range[0], attendance_range[1], 10)
            
            predictions_grid = []
            for h in hours_grid:
                for a in attendance_grid:
                    pred = model.predict([[h, a]])[0]
                    predictions_grid.append({
                        'Hours': h,
                        'Attendance': a,
                        'Predicted_Score': pred
                    })
            
            grid_df = pd.DataFrame(predictions_grid)
            
            st.dataframe(grid_df.head(20), use_container_width=True)
    
    with col2:
        st.subheader("🎯 Live Prediction Surface")
        
        # Create prediction surface
        hours_surface = np.linspace(1, 12, 20)
        attendance_surface = np.linspace(40, 100, 20)
        H, A = np.meshgrid(hours_surface, attendance_surface)
        
        # Predict for surface
        surface_predictions = []
        for i in range(len(hours_surface)):
            for j in range(len(attendance_surface)):
                pred = model.predict([[H[j,i], A[j,i]]])[0]
                surface_predictions.append(pred)
        
        Z = np.array(surface_predictions).reshape(H.shape)
        
        # Create 3D surface plot
        fig_surface = go.Figure(data=[
            go.Surface(
                x=hours_surface,
                y=attendance_surface,
                z=Z,
                colorscale='viridis',
                opacity=0.8
            )
        ])
        
        fig_surface.update_layout(
            title='Prediction Surface: Score vs Hours & Attendance',
            scene=dict(
                xaxis_title='Hours Studied',
                yaxis_title='Attendance %',
                zaxis_title='Predicted Score'
            ),
            height=500
        )
        
        st.plotly_chart(fig_surface, use_container_width=True)

with tab5:
    st.header("📋 Data Insights & Recommendations")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🔍 Key Insights")
        
        # Calculate insights
        correlation_hours = data['Hours_Studied'].corr(data['Final_Score'])
        correlation_attendance = data['Attendance'].corr(data['Final_Score'])
        
        insights = [
            f"📖 **Study Hours Impact**: {correlation_hours:.3f} correlation with final score",
            f"📅 **Attendance Impact**: {correlation_attendance:.3f} correlation with final score",
            f"🎯 **Model Accuracy**: {metrics['r2_score']:.1%} of score variance explained",
            f"📊 **Average Error**: ±{metrics['mae']:.1f} points in predictions",
            f"🏆 **Best Strategy**: {'High study hours' if abs(model.coef_[0]) > abs(model.coef_[1]) else 'High attendance'} has stronger impact"
        ]
        
        for insight in insights:
            st.markdown(insight)
        
        st.subheader("💡 Recommendations")
        recommendations = [
            "🎯 **For Low Performers**: Focus on increasing study hours first",
            "📅 **Attendance Matters**: Maintain >80% attendance for optimal results",
            "⚖️ **Balance is Key**: Combine good attendance with adequate study time",
            "📈 **Growth Target**: 1 extra hour of study ≈ 8 point score increase",
            "🔄 **Regular Monitoring**: Track both metrics consistently"
        ]
        
        for rec in recommendations:
            st.markdown(rec)
    
    with col2:
        st.subheader("📊 Performance Categories")
        
        # Categorize students
        data_copy = data.copy()
        data_copy['Performance_Category'] = pd.cut(
            data_copy['Final_Score'], 
            bins=[0, 60, 75, 85, 100], 
            labels=['Needs Improvement', 'Fair', 'Good', 'Excellent']
        )
        
        category_counts = data_copy['Performance_Category'].value_counts()
        
        fig_pie = px.pie(
            values=category_counts.values,
            names=category_counts.index,
            title="Student Performance Distribution"
        )
        st.plotly_chart(fig_pie, use_container_width=True)
        
        # Study habits analysis
        st.subheader("📚 Study Habits Analysis")
        
        avg_by_category = data_copy.groupby('Performance_Category').agg({
            'Hours_Studied': 'mean',
            'Attendance': 'mean'
        }).round(2)
        
        st.dataframe(avg_by_category, use_container_width=True)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666;">
    <p>📚 Student Score Prediction Dashboard | Built with Streamlit & Python</p>
    <p>🤖 Machine Learning Model: Linear Regression | 📊 Data Visualization: Plotly</p>
    <p> Project by Mahender Banoth ( IIT Patna ) </p>
</div>
""", unsafe_allow_html=True)
