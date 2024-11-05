import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='Compute cosine similarity between two CSV files')
    parser.add_argument('--csv1', required=True, help='Path to the first CSV file')
    parser.add_argument('--csv2', required=True, help='Path to the second CSV file')
    args = parser.parse_args()
    return args

def main():
    args = parse_args()

    df1 = pd.read_csv(args.csv1)
    df2 = pd.read_csv(args.csv2)

    columns_mapping = {
        'Aesthetic': 'aesthetics',
        'Happy': 'happy',
        'Natural': 'natural',
        'New': 'new',
        'Scary': 'scary',
        'Complex': 'complex'
    }

    existing_columns_1 = [col for col in columns_mapping.keys() if col in df1.columns]
    existing_columns_2 = [columns_mapping[col] for col in existing_columns_1]

    similarities = []
    for index, row1 in df1.iterrows():
        row2 = df2.iloc[index]
        values1 = row1[existing_columns_1].values.reshape(1, -1)
        values2 = row2[existing_columns_2].values.reshape(1, -1)
        similarity = cosine_similarity(values1, values2)[0][0]
        similarities.append({
            'image_file': row1['image_file'],
            'cosine_similarity': similarity
        })

    similarity_df = pd.DataFrame(similarities)
    similarity_df.to_csv('cosine_similarities.csv', index=False)

    average_similarity = similarity_df['cosine_similarity'].mean()
    print(f"Cosine similarities saved to 'cosine_similarities.csv'")
    print(f"Average Cosine Similarity: {average_similarity:.4f}")

if __name__ == '__main__':
    main()
