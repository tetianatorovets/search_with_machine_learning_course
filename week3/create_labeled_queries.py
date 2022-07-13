import os
import argparse
import xml.etree.ElementTree as ET
import pandas as pd
import numpy as np
import csv

# Useful if you want to perform stemming.
import nltk
stemmer = nltk.stem.PorterStemmer()

categories_file_name = r'/workspace/datasets/product_data/categories/categories_0001_abcat0010000_to_pcmcat99300050000.xml'

queries_file_name = r'/workspace/datasets/train.csv'
output_file_name = r'/workspace/datasets/labeled_query_data.txt'

parser = argparse.ArgumentParser(description='Process arguments.')
general = parser.add_argument_group("general")
general.add_argument("--min_queries", default=1,  help="The minimum number of queries per category label (default is 1)")
general.add_argument("--output", default=output_file_name, help="the file to output to")

args = parser.parse_args()
output_file_name = args.output

if args.min_queries:
    min_queries = int(args.min_queries)

# The root category, named Best Buy with id cat00000, doesn't have a parent.
root_category_id = 'cat00000'

tree = ET.parse(categories_file_name)
root = tree.getroot()

# Parse the category XML file to map each category id to its parent category id in a dataframe.
categories = []
parents = []
for child in root:
    id = child.find('id').text
    cat_path = child.find('path')
    cat_path_ids = [cat.find('id').text for cat in cat_path]
    leaf_id = cat_path_ids[-1]
    if leaf_id != root_category_id:
        categories.append(leaf_id)
        parents.append(cat_path_ids[-2])
parents_df = pd.DataFrame(list(zip(categories, parents)), columns =['category', 'parent'])

# Read the training data into pandas, only keeping queries with non-root categories in our category tree.
df = pd.read_csv(queries_file_name)[['category', 'query']]
df = df[df['category'].isin(categories)]

# IMPLEMENT ME: Convert queries to lowercase, and optionally implement other normalization, like stemming.
df['query_normalised'] = df['query'].str.lower()
df['query_normalised'] = df['query_normalised'].apply(lambda x: x.replace('"', ''))
df['query_normalised'] = df['query_normalised'].apply(lambda x: x.replace(r'\s+', ' '))
# df['query_normalised'] = df['query_normalised'].apply(lambda x: [stemmer.stem(y) for y in x])
# print(df.head())
# IMPLEMENT ME: Roll up categories to ancestors to satisfy the minimum number of queries per category.

df['category_agg'] = df['category']
threshold = 10000
while (df.groupby('category_agg')['query_normalised'].count().min()) < threshold:
    df_agg = df.groupby('category_agg')['query_normalised'].count().reset_index()
    df_agg = df_agg.merge(parents_df, how = 'inner', left_on =  df_agg.category_agg, right_on= parents_df.category)
    df_agg['category_agg'] = np.where(df_agg['query_normalised'] < threshold, df_agg['parent'], df_agg['category'])
    df = df.merge(df_agg[['category_agg', 'category']], how = 'inner', on = ['category'])
    df['category_agg'] = df['category_agg_y']
    df = df.drop(columns=['category_agg_x', 'category_agg_y'])


# Create labels in fastText format.
df['label'] = '__label__' + df['category']

# Output labeled query data as a space-separated file, making sure that every category is in the taxonomy.
df = df[df['category'].isin(categories)]
df['output'] = df['label'] + ' ' + df['query']
df[['output']].to_csv(output_file_name, header=False, sep='|', escapechar='\\', quoting=csv.QUOTE_NONE, index=False)
