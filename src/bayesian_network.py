import csv
import pickle
import networkx as nx
import pandas as pd

from matplotlib import pyplot as plt
from pgmpy.models import BayesianNetwork
from pgmpy.inference import VariableElimination
from pgmpy.estimators import MaximumLikelihoodEstimator, HillClimbSearch, K2Score
from sklearn.preprocessing import KBinsDiscretizer
from utils import reduce_categories


# Create a Bayesian Network model
def create_bayesian_network(dataset):
    edges = []
    columns_to_exclude = ['MPAA', 'Country', 'Budget', 'Worldwide Gross', 'Genre', 'Score', 'Votes']

    # Define the edges of the Bayesian Network
    for column in dataset.columns:
        if column not in columns_to_exclude and column != 'Worldwide Gross':
            edges.append(('Worldwide Gross', column))

        edges.append(('Genre', 'MPAA'))
        edges.append(('Genre', 'Score'))
        edges.append(('Genre', 'Votes'))
        edges.append(('Company', 'Budget'))
        edges.append(('Company', 'Country'))
        edges.append(('Director', 'Genre'))
        edges.append(('Writer', 'Genre'))
        edges.append(('Main Actor', 'Genre'))

    # Perform a Hill Climbing Search to estimate the Bayesian Network  
    # hc_k2 = HillClimbSearch(dataset)
    # k2_model =  hc_k2.estimate(scoring_method=K2Score(dataset), max_iter=100)
    # edges = k2_model.edges()

    # Create the Bayesian Network
    bn = BayesianNetwork(edges)
    bn.fit(dataset, estimator=MaximumLikelihoodEstimator, n_jobs=2)

    with open('../resources/bayesian_networks/bayesian_network.pkl', 'wb') as output:
        pickle.dump(bn, output)

    return bn

# Load the Bayesian Network model
def load_bayesian_network():
    with open('../resources/bayesian_networks/bayesian_network.pkl', 'rb') as file:
        model = pickle.load(file)
    
    return model

# Display the Bayesian Network
def display_bayesian_network(bn: BayesianNetwork):
    G = nx.MultiDiGraph(bn.edges())
    pos = nx.spring_layout(G, iterations=100, k=2, threshold=5, pos=nx.spiral_layout(G))
    
    nx.draw_networkx_nodes(G, pos, node_size=250, node_color="#ff574c")
    nx.draw_networkx_labels(
        G,
        pos,
        font_size=10,
        font_weight="bold",
        clip_on=True,
        horizontalalignment="center",
        verticalalignment="bottom",
    )
    nx.draw_networkx_edges(
        G,
        pos,
        arrows=True,
        arrowsize=8,
        arrowstyle="->",
        edge_color="blue",
        connectionstyle="arc3,rad=0.2",
        min_source_margin=1.2,
        min_target_margin=1.5,
        edge_vmin=2,
        edge_vmax=2,
    )

    plt.title("Bayesian Network Graph")
    plt.savefig('../resources/plots/bayesian_networks/bayesian_network.png')
    plt.show()
    plt.clf()

# Show the Conditional Probability Distributions
def show_CPD(bn: BayesianNetwork):
    cpds = bn.get_cpds()

    for cpd in cpds:
        if cpd.variable != 'Votes':
            print(f'CPD of {cpd.variable}:')
            print(cpd, '\n')

# Generate a random example
def generate_random_example(bn: BayesianNetwork):
    return bn.simulate(n_samples=1).drop(columns=['Worldwide Gross'], axis=1)


# Get the dataset
with open('../resources/dataset/Movie_dataset_cleaned.csv', mode='r', encoding='utf-8-sig') as movieCsv:
    reader = csv.DictReader(movieCsv)
    dataset = list(reader)
    df = pd.DataFrame(dataset)

df.drop(columns=['Title'], axis=1, inplace=True)

# Pre-process the data
df['Budget'] = pd.to_numeric(df['Budget'], errors='coerce')
df['Worldwide Gross'] = pd.to_numeric(df['Worldwide Gross'], errors='coerce')
df['Score'] = pd.to_numeric(df['Score'], errors='coerce')
df['Votes'] = pd.to_numeric(df['Votes'], errors='coerce')
df['Runtime'] = pd.to_numeric(df['Runtime'], errors='coerce')
df['Year'] = pd.to_numeric(df['Year'], errors='coerce')

# Ensure categorical variables are treated as such
categorical_columns = ['Genre', 'MPAA', 'Company', 'Country', 'Director', 'Writer', 'Main Actor']
for col in categorical_columns:
    df[col] = df[col].astype('category')

# Discretize continuous variables
discretizer = KBinsDiscretizer(n_bins=10, encode='ordinal', strategy='uniform')
df['Budget'] = discretizer.fit_transform(df[['Budget']])
df['Runtime'] = discretizer.fit_transform(df[['Runtime']])
df['Worldwide Gross'] = discretizer.fit_transform(df[['Worldwide Gross']])

# Reduce the number of categories in categorical variables
reduce_categories(df, 'Company', threshold=10)
reduce_categories(df, 'Director', threshold=20)
reduce_categories(df, 'Writer', threshold=10)
reduce_categories(df, 'Main Actor', threshold=20)


bn = create_bayesian_network(df)
# bn = load_bayesian_network()
display_bayesian_network(bn)
show_CPD(bn)

# Generate a random example
example = generate_random_example(bn)
print("Example: " + str(example))

inference = VariableElimination(bn)
result = inference.query(variables=['Worldwide Gross'], evidence=example.iloc[0].to_dict())
print(result)