import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def load_organizer_user_data():
    # Load and preprocess organizer user data
    org_user = pd.read_csv('data/organizer_user_data.csv')
    org_user = org_user[org_user['organizer id'] != -1]  # Drop test data with organizer id = -1
    org_user.rename(
        columns={'organizer id': 'org_id', 'Unnamed: 0': 'count', 'user id': 'user_id', 'issued time': 'Timestamp',
                 'districts': 'Location'}, inplace=True)
    org_user = org_user.drop(['count'], axis=1)
    _mapping = org_user.Location.unique()
    _mapping_dict = dict(zip(_mapping, range(len(_mapping))))
    _org_user = org_user.apply(lambda col: col.map(_mapping_dict) if col.name == 'Location' else col)
    _org_user = _org_user.rename(columns={'Timestamp': 'dates'})
    _org_user['ts'] = _org_user['dates'].apply(lambda x: pd.Timestamp(x))
    _org_user['Timestamp'] = _org_user.ts.values.astype(np.int64) // 10 ** 9
    _org_user = _org_user.drop(['dates'], axis=1)
    _org_user = _org_user.drop(['ts'], axis=1)
    _org_user = _org_user.rename(columns={'user_id': 'UserId', 'org_id': 'ItemId'})
    _org_user = _org_user[['UserId', 'ItemId', 'Location', 'Timestamp']]
    return _org_user

def create_volunteer_network():
    # Set up the network based on volunteer interactions
    organizer_user_data = load_organizer_user_data()
    network = pd.crosstab(organizer_user_data['UserId'], organizer_user_data['ItemId'])
    network = network.apply(lambda row: row / row.sum(), axis=1)  # Normalize probability P(o|v)
    
    row_ids = network.index
    cosine_similarity_matrix = cosine_similarity(network)
    np.fill_diagonal(cosine_similarity_matrix, -1)
    
    neighbor_df = pd.DataFrame(cosine_similarity_matrix, index=row_ids, columns=row_ids)
    neighbor_df_sorted = pd.DataFrame(
        neighbor_df.apply(lambda x: list(neighbor_df.columns[np.array(x).argsort()[::-1][:10]]), axis=1)
        .to_list(), columns=['Neighbor_1', 'Neighbor_2', 'Neighbor_3', 'Neighbor_4', 'Neighbor_5', 
                             'Neighbor_6', 'Neighbor_7', 'Neighbor_8', 'Neighbor_9', 'Neighbor_10'])
    
    neighbor_df_sorted['UserId'] = row_ids
    neighbor_melted = pd.melt(neighbor_df_sorted, id_vars=['UserId'])
    neighbor_melted['Weight'] = 1
    neighbor_melted = neighbor_melted.rename(columns={'UserId': 'Followee', 'value': 'Follower'})
    volunteer_network = neighbor_melted[['Follower', 'Followee', 'Weight']]
    
    volunteer_network.to_csv('volunteer_network.tsv')

if __name__ == '__main__':
    create_volunteer_network()
