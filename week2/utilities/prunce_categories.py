import pandas as pd

path_original_file = "/workspace/datasets/fasttext/labeled_products.txt"
path_new_file = "/workspace/datasets/fasttext/pruned_labeled_products.txt"

df_original = pd.read_table(path_original_file, names = ['text'])
df_original[['category', 'title']] = df_original["text"].str.split(" ", 1, expand=True)
df_grouped = df_original.groupby('category')['title'].count().reset_index()
df_grouped = df_grouped[df_grouped['title'] >= 500]
df_final = df_original.merge(df_grouped, how = 'left', on = 'category')
df_final = df_final[~pd.isnull(df_final['title_y'])]
df_final['text'].to_csv(path_new_file, header=None, index=None)
print('done')