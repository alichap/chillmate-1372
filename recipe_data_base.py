import pandas as pd
import numpy as np

recipe_path_recipe_database = 'raw_data/Food Ingredients and Recipe Dataset with Image Name Mapping.csv'
food_list_path = 'raw_data/food_ingredients_and_allergens.csv'

recipe_df= pd.read_csv(recipe_path_recipe_database)
list_food_df= pd.read_csv(food_list_path)


def dict_data_base_setup():
    '''
    Function to create a dictionary with key as ingredient ans value as list of recipe containe key's ingredient
    '''
    recipe_df['Cleaned_Ingredients_split'] = recipe_df['Cleaned_Ingredients'].apply(lambda x: x.split() if isinstance(x, str) else x)
    list_food = list(set(list_food_df['Main Ingredient'].to_list()))
    lowercase_list_ing = [item.lower() for item in list_food]

    def find_common_elements(row):
        return list(set(row).intersection(lowercase_list_ing))

    recipe_df['ingredients_clean'] = recipe_df['Cleaned_Ingredients_split'].apply(lambda x: find_common_elements(x))

    recipe_df_clean_filtered = recipe_df[recipe_df['ingredients_clean'].apply(lambda x: len(x) > 0)]

    df_expanded = recipe_df_clean_filtered.explode('ingredients_clean')
    grouped_df = df_expanded.groupby('ingredients_clean')['Title'].agg(list).reset_index()
    dict_recipe= dict(zip(grouped_df['ingredients_clean'],grouped_df['Title']))

    #print(dict_recipe)
    return dict_recipe

#print(dict_data_base_setup().keys())
