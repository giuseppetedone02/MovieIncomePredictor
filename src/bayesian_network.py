import pickle
import networkx as nx
import pandas as pd

from matplotlib import pyplot as plt
from pgmpy.models import BayesianNetwork
from pgmpy.inference import VariableElimination
from pgmpy.estimators import MaximumLikelihoodEstimator, HillClimbSearch, K2Score
from sklearn.preprocessing import KBinsDiscretizer


# Create a Bayesian Network model
def create_bayesian_network(dataset):
    edges = []
    columns_to_exclude = ['MPAA', 'Country', 'Budget', 'ROI', 'Worldwide Gross', 'Genre', 'Score', 'Votes']

    # Define the edges of the Bayesian Network
    for column in dataset.columns:
        if column not in columns_to_exclude and column != 'Worldwide Gross':
            edges.append(('Worldwide Gross', column))

        edges.append(('Genre', 'MPAA'))
        edges.append(('Genre', 'Score'))
        edges.append(('Genre', 'Votes'))
        edges.append(('Company', 'Budget'))
        edges.append(('Company', 'Country'))
        edges.append(('Director_Num_Movies', 'Genre'))
        edges.append(('Writer_Num_Movies', 'Genre'))
        edges.append(('Main_Actor_Num_Movies', 'Genre'))
        edges.append(('Budget', 'ROI'))

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

    with open('../resources/logs/bayesian_network/cpd.txt', 'w') as logfile:
        for cpd in cpds:
            if cpd.variable != 'Votes':
                logfile.write(f'CPD of {cpd.variable}:\n')
                logfile.write(str(cpd) + '\n\n')

# Generate a random example
def generate_random_example(bn: BayesianNetwork):
    return bn.simulate(n_samples=1).drop(columns=['Worldwide Gross'], axis=1)


# Get the dataset
df = pd.read_csv('../resources/dataset/Movie_dataset_with_prolog_results.csv', encoding='utf-8-sig')

# Ensure categorical variables are treated as such
categorical_columns = ['Genre', 'MPAA', 'Company', 'Country']
for col in categorical_columns:
    df[col] = df[col].astype('category')

# Discretize continuous variables
discretizer = KBinsDiscretizer(n_bins=10, encode='ordinal', strategy='uniform')
df['Budget'] = discretizer.fit_transform(df[['Budget']])
df['Runtime'] = discretizer.fit_transform(df[['Runtime']])
df['Worldwide Gross'] = discretizer.fit_transform(df[['Worldwide Gross']])

# Reduce the number of categories in categorical variables
df['Director_Num_Movies'] = df['Director'].map(df['Director'].value_counts())
df['Writer_Num_Movies'] = df['Writer'].map(df['Writer'].value_counts())
df['Main_Actor_Num_Movies'] = df['Main Actor'].map(df['Main Actor'].value_counts())

df.drop(columns=['Title', 'Director', 'Writer', 'Main Actor'], axis=1, inplace=True)

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

# Query the Bayesian Network with a film with MPAA of NC-17
result = inference.query(variables=['Worldwide Gross'], evidence={'MPAA': 'NC-17'})
print('\nCPD per film con MPAA = NC-17: \n', result)