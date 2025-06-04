import pandas as pd
import numpy as np
from sksurv.ensemble import ComponentwiseGradientBoostingSurvivalAnalysis as CGBSA
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sksurv.metrics import cumulative_dynamic_auc



starting_RBS = ['James Cook', "De'Von Achane",'Rhamondre Stevenson','Breece Hall',
                'Derrick Henry','Chase Brown','Jerome Ford','Najee Harris',
                'Joe Mixon','Jonathan Taylor','Travis Etienne','Tony Pollard',
                'Kareem Hunt','J.K. Dobbins',
                'Rico Dowdle','Tyrone Tracy Jr.','Saquon Barkley','Brian Robinson',
                "D'Andre Swift",'Jahmyr Gibbs','Josh Jacobs','Aaron Jones',
                'Bijan Robinson','Chuba Hubbard','Alvin Kamara','Bucky Irving',
                'James Conner','Kyren Williams','Jordan Mason','Kenneth Walker III']

def find_missing_players(file_path='scaled_RB_data.csv'):
    """
    Identifies players in starting_RBS list that are not in the dataset.

    Parameters:
    file_path (str): Path to the CSV file containing the data

    Returns:
    list: Names of players not found in the dataset
    """
    # Load the data
    df = pd.read_csv(file_path)

    # Fix the syntax error in starting_RBS (extra comma after 'Tony Pollard')
    fixed_starting_RBS = [name for name in starting_RBS if name is not None]

    # Get all player names from the dataset
    all_players = set(df['Player Name'].unique())

    # Find players in starting_RBS not in the dataset
    missing_players = [name for name in fixed_starting_RBS if name not in all_players]

    # Print the missing players
    if missing_players:
        print("Players in starting_RBS not found in dataset:")
        for player in missing_players:
            print(f"- {player}")
    else:
        print("All players in starting_RBS are in the dataset.")

    return missing_players

# Call the function to check for missing players
missing_players = find_missing_players()

#Survival Modelling
def train_survival_model(file_path='scaled_RB_data.csv'):
    """
    Train a scikit-survival model on running back data.
    
    Parameters:
    file_path (str): Path to the CSV file containing the data
    
    Returns:
    tuple: Trained model and test data
    """
    # Load the data
    df = pd.read_csv(file_path)
    # Print how many rows contain a NaN value
 

    # Drop rows with NaN values
    df = df.fillna(0)  # Fill NaN values with 0

    df = df[df['Year'] >= 2024] #Remove rookies

    # Remove specified columns
    if 'PB' in df.columns and 'Player Name' in df.columns and 'Year' in df.columns:
        X = df.drop(['PB', 'Player Name', 'Year', 'St','Player'], axis=1)
    else:
        raise ValueError("Required columns not found in the dataset")
    
    # Prepare the survival target (event, time)
    # Assuming 'St' represents the time lived (seasons played)
    time = df['St'].values
    
    # For demonstration, we'll assume an event occurred for all players
    # In a real scenario, you might have a column indicating if a career ended (event=True) or is ongoing (event=False)
    # Set event to True if player is not in starting_RBS (career ended/not active starter)
    # and False if they are in the list (active starter, censored data)
    event = np.array([False if name in starting_RBS else True for name in df['Player Name']])
    
    # Create structured array for survival analysis
    y = np.array(list(zip(event, time)), dtype=[('Status', '?'), ('Survival_time', '<f8')])
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
    
    # Train a Random Survival Forest model
    cgbsa = CGBSA(n_estimators=1000, random_state=42)
    
    cgbsa.fit(X_train, y_train)
    
    return cgbsa, X_test, y_test

def plot_survival_distribution(model, X_test, y_test):
    """
    Plot the cumulative distribution function for the survival model.
    
    Parameters:
    model: Trained survival model
    X_test: Test features
    y_test: Test targets
    """
    import matplotlib.pyplot as plt
    
    # Predict survival function for test data
    surv_funcs = model.predict_survival_function(X_test)
    
    # Create a figure for plotting
    plt.figure(figsize=(12, 6))
    
    # Plot survival curves for each test instance
    for i, surv_func in enumerate(surv_funcs):
        # Only plot a subset of curves to avoid overcrowding
        if i % 5 == 0:  # Plot every 5th curve
            plt.step(surv_func.x, surv_func(surv_func.x), where="post", 
                        label=f"Test sample {i}" if i < 25 else "", alpha=0.3)
    
    # Calculate and plot the mean survival curve
    mean_times = surv_funcs[0].x
    mean_survival = np.zeros_like(mean_times, dtype=float)
    
    for surv_func in surv_funcs:
        mean_survival += surv_func(mean_times)
    mean_survival /= len(surv_funcs)
    
    plt.step(mean_times, mean_survival, where="post", 
                color="red", linewidth=2, label="Mean survival")
    
    # Plot formatting
    plt.xlabel("Seasons (Starts)")
    plt.ylabel("Survival Probability")
    plt.title("RB Survival Probability by Seasons Started")
    plt.ylim(0, 1)
    plt.grid(True)
    
    # Add legend for the first few and the mean
    handles, labels = plt.gca().get_legend_handles_labels()
    if len(handles) > 10:
        handles = handles[:5] + [handles[-1]]
        labels = labels[:5] + [labels[-1]]
    plt.legend(handles, labels, loc="best")
    
    plt.tight_layout()
    plt.show()

# Train the model and get test data
model, X_test, y_test = train_survival_model()

# Plot the survival distribution
plot_survival_distribution(model, X_test, y_test)