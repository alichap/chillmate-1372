from recipe_data_base  import *

#list_cnn = ['spinach','chicken'] # to change

def find_common_recipe(list_ing,dict_recipe):
    dict_recipe = dict_data_base_setup()
    list_ing_separate = [dict_recipe.get(key) for key in list_ing]
    if not list_ing_separate:
        return []
# Initialize with the first list
    intersection_result = set(list_ing_separate[0])
# Iteration with all ingredients
    for i in list_ing_separate [1:]:
        common_recipe = intersection_result.intersection(i)
    #print (common_recipe)
    #print(len(common_recipe))

    return common_recipe

#find_common_recipe(list_cnn)
