import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from scipy.spatial.distance import cdist
import os

# Ensure the results directory exists
result_dir = "results"
os.makedirs(result_dir, exist_ok=True)

def generate_ellipsoid_clusters(distance, n_samples=100, cluster_std=0.5):
    """
    Generates two ellipsoid clusters with the second cluster shifted by a specified distance.

    Parameters:
    - distance (float): The distance to shift the second cluster along both x and y axes.
    - n_samples (int): Number of samples per cluster.
    - cluster_std (float): Standard deviation of the clusters.

    Returns:
    - X (ndarray): Combined feature data.
    - y (ndarray): Combined labels.
    """
    np.random.seed(0)
    covariance_matrix = np.array([[cluster_std, cluster_std * 0.8], 
                                  [cluster_std * 0.8, cluster_std]])
    
    # Generate the first cluster (class 0)
    X1 = np.random.multivariate_normal(mean=[1, 1], cov=covariance_matrix, size=n_samples)
    y1 = np.zeros(n_samples)

    # Generate the second cluster (class 1)
    X2 = np.random.multivariate_normal(mean=[1, 1], cov=covariance_matrix, size=n_samples)
    
    # Shift the second cluster along both axes by 'distance'
    X2 += np.array([distance, distance])
    
    y2 = np.ones(n_samples)

    # Combine the clusters into one dataset
    X = np.vstack((X1, X2))
    y = np.hstack((y1, y2))
    return X, y

# Function to fit logistic regression and extract coefficients
def fit_logistic_regression(X, y):
    """
    Fits a logistic regression model to the data and extracts the coefficients.

    Parameters:
    - X (ndarray): Feature data.
    - y (ndarray): Labels.

    Returns:
    - model (LogisticRegression): Fitted logistic regression model.
    - beta0 (float): Intercept of the model.
    - beta1 (float): Coefficient for feature x1.
    - beta2 (float): Coefficient for feature x2.
    """
    model = LogisticRegression()
    model.fit(X, y)
    beta0 = model.intercept_[0]
    beta1, beta2 = model.coef_[0]
    return model, beta0, beta1, beta2

def do_experiments(start, end, step_num):
    """
    Conducts experiments by shifting clusters, fitting logistic regression models,
    and analyzing how model parameters and performance metrics change with shift distance.

    Parameters:
    - start (float): Starting value of shift distance.
    - end (float): Ending value of shift distance.
    - step_num (int): Number of shift distances to evaluate.
    """
    # Define the range of shift distances
    shift_distances = np.linspace(start, end, step_num)  # Range of shift distances
    beta0_list, beta1_list, beta2_list, slope_list, intercept_list, loss_list, margin_widths = [], [], [], [], [], [], []
    sample_data = {}  # Store sample datasets and models for visualization

    n_samples = 8
    n_cols = 2  # Fixed number of columns for subplot
    n_rows = (n_samples + n_cols - 1) // n_cols  # Calculate rows needed
    plt.figure(figsize=(20, n_rows * 10))  # Adjust figure height based on rows

    # Run experiments for each shift distance
    for i, distance in enumerate(shift_distances, 1):
        X, y = generate_ellipsoid_clusters(distance=distance)
        
        # Implement: record all necessary information for each distance
        model, beta0, beta1, beta2 = fit_logistic_regression(X, y)
        beta0_list.append(beta0)
        beta1_list.append(beta1)
        beta2_list.append(beta2)
         # Implement: Calculate decision boundary slope and intercept
        slope = -beta1 / beta2
        intercept = -beta0 / beta2
        slope_list.append(slope)
        intercept_list.append(intercept)
        # Implement: Calculate and store logistic loss
        probabilities = model.predict_proba(X)[:, 1]
        logistic_loss = -np.mean(y * np.log(probabilities) + (1 - y) * np.log(1 - probabilities))
        loss_list.append(logistic_loss)
        
        # Implement: Plot the dataset
        plt.subplot(n_rows, n_cols, i)
        plt.scatter(X[y == 0, 0], X[y == 0, 1], color='blue', label='Class 0', alpha=0.6)
        plt.scatter(X[y == 1, 0], X[y == 1, 1], color='red', label='Class 1', alpha=0.6)
        
     
        
        # Calculate margin width between 70% confidence contours for each class
        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200), np.linspace(y_min, y_max, 200))
        Z = model.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]
        Z = Z.reshape(xx.shape)

       
        
        # Plot the decision boundary
        x_vals = np.array([x_min, x_max])
        y_vals = slope * x_vals + intercept
        plt.plot(x_vals, y_vals, '--', color='green', label='Decision Boundary')
        
        # Calculate margin width using the inverse of the norm of the coefficients
        margin_width = 1 / np.sqrt(beta1**2 + beta2**2)
        margin_widths.append(margin_width)
        
        # Plot fading red and blue contours for confidence levels
        contour_levels = [0.7, 0.8, 0.9]
        alphas = [0.05, 0.1, 0.15]  # Increasing opacity for higher confidence levels
        for level, alpha in zip(contour_levels, alphas):
            # Fading red for Class 1
            plt.contourf(xx, yy, Z, levels=[level, 1.0], colors=['red'], alpha=alpha)
            # Fading blue for Class 0
            plt.contourf(xx, yy, Z, levels=[0.0, 1 - level], colors=['blue'], alpha=alpha)
        
        plt.title(f"Shift Distance = {distance:.2f}", fontsize=16)
        plt.xlabel("x1")
        plt.ylabel("x2")
        
        # Display decision boundary equation and margin width on the plot
        equation_text = f"{beta0:.2f} + {beta1:.2f} * x1 + {beta2:.2f} * x2 = 0\nx2 = {slope:.2f} * x1 + {intercept:.2f}"
        margin_text = f"Margin Width: {margin_width:.2f}"
        plt.text(x_min + 0.1, y_max - 1.0, equation_text, fontsize=10, color="black", ha='left',
                 bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.3'))
        plt.text(x_min + 0.1, y_max - 1.5, margin_text, fontsize=10, color="black", ha='left',
                 bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.3'))
        
        # Add legend only to the first subplot to avoid repetition
        if i == 1:
            plt.legend(loc='lower right', fontsize=12)
        
        # Store sample data for potential further analysis or visualization
        sample_data[distance] = (X, y, model, beta0, beta1, beta2, margin_width)

    plt.tight_layout()
    plt.savefig(f"{result_dir}/dataset.png")
    plt.close()

    # Plot 1: Parameters vs. Shift Distance
    plt.figure(figsize=(18, 15))

    # Implement: Plot beta0
    plt.subplot(3, 3, 1)
    plt.plot(shift_distances, beta0_list, marker='o', linestyle='-', color='blue')
    plt.title("Shift Distance vs Beta0")
    plt.xlabel("Shift Distance")
    plt.ylabel("Beta0")
    plt.grid(True)

    # Implement: Plot beta1
    plt.subplot(3, 3, 2)
    plt.plot(shift_distances, beta1_list, marker='o', linestyle='-', color='orange')
    plt.title("Shift Distance vs Beta1 (Coefficient for x1)")
    plt.xlabel("Shift Distance")
    plt.ylabel("Beta1")
    plt.grid(True)

    # Implement: Plot beta2
    plt.subplot(3, 3, 3)
    plt.plot(shift_distances, beta2_list, marker='o', linestyle='-', color='green')
    plt.title("Shift Distance vs Beta2 (Coefficient for x2)")
    plt.xlabel("Shift Distance")
    plt.ylabel("Beta2")
    plt.grid(True)

    # Implement: Plot beta1 / beta2 (Slope)
    slope_ratio = np.array(beta1_list) / np.array(beta2_list)
    plt.subplot(3, 3, 4)
    plt.plot(shift_distances, slope_ratio, marker='o', linestyle='-', color='red')
    plt.title("Shift Distance vs Beta1 / Beta2 (Slope)")
    plt.xlabel("Shift Distance")
    plt.ylabel("Beta1 / Beta2")
    plt.ylim(-2, 0)
    plt.grid(True)

    # Implement: Plot beta0 / beta2 (Intercept ratio)
    intercept_ratio = np.array(beta0_list) / np.array(beta2_list)
    plt.subplot(3, 3, 5)
    plt.plot(shift_distances, intercept_ratio, marker='o', linestyle='-', color='purple')
    plt.title("Shift Distance vs Beta0 / Beta2 (Intercept Ratio)")
    plt.xlabel("Shift Distance")
    plt.ylabel("Beta0 / Beta2")
    plt.grid(True)

    # Plot logistic loss
    plt.subplot(3, 3, 6)
    plt.plot(shift_distances, loss_list, marker='o', linestyle='-', color='brown')
    plt.title("Shift Distance vs Logistic Loss")
    plt.xlabel("Shift Distance")
    plt.ylabel("Logistic Loss")
    plt.grid(True)

    # Implement: Plot margin width
    plt.subplot(3, 3, 7)
    plt.plot(shift_distances, margin_widths, marker='o', linestyle='-', color='cyan')
    plt.title("Shift Distance vs Margin Width")
    plt.xlabel("Shift Distance")
    plt.ylabel("Margin Width")
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(f"{result_dir}/parameters_vs_shift_distance.png")
    plt.close()

    print("Experiments completed successfully!")
    print(f"Results saved in the '{result_dir}' directory.")

if __name__ == "__main__":
    start = 0.25
    end = 2.0
    step_num = 8
    do_experiments(start, end, step_num)
