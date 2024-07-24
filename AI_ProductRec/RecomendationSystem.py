import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import LabelEncoder

# Load purchase history and product information from CSV files
purchase_history = pd.read_csv('grocery_sells.csv')
product_info = pd.read_csv('Online_Retail_Categorized.csv')

# Merge purchase history with product information
merged_data = pd.merge(purchase_history, product_info, left_on=['Category', 'Sub Category'], right_on=['Category', 'Sub Category'], how='inner')

# Create a user-product matrix
user_product_matrix = pd.pivot_table(merged_data, values='Quantity', index='Customer Name', columns=['Category', 'Sub Category'], fill_value=0)

# Flatten multi-index columns
user_product_matrix.columns = [' '.join(map(str, col)).strip() for col in user_product_matrix.columns.values]

# Convert product names to numerical labels using LabelEncoder
le = LabelEncoder()
user_product_matrix.columns = le.fit_transform(user_product_matrix.columns)

# Calculate the cosine similarity between users
user_similarity = cosine_similarity(user_product_matrix)
def get_recommendations(customer_name, num_recommendations=5):

    user_index = user_product_matrix.index.get_loc(customer_name)
    sim_scores = list(enumerate(user_similarity[user_index]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:num_recommendations + 1]
    product_indices = [i[0] for i in sim_scores]

    # Ensure that the indices are within the range of labels encountered during fitting
    if max(product_indices) >= len(le.classes_):
        print("Some indices are out of range. Adjusting indices.")
        product_indices = [idx % len(le.classes_) for idx in product_indices]

    # Use the same LabelEncoder for inverse transforming
    recommended_products = le.inverse_transform(product_indices)
    return recommended_products


while True:
    # Example: Get recommendations for a specific user (replace 'customer_name' with an actual customer name)
    customer_name = input("Enter customer name (type 'quit' or 'exit' to stop): ")
    
    # Check if the user wants to quit
    if customer_name.lower() in ['quit', 'exit']:
        print("Exiting recommendation system.")
        break
    
    recommendations = get_recommendations(customer_name)
    
    if len(recommendations) > 0:
        print(f"Recommendations for Customer {customer_name}:", recommendations)
    else:
        print(f"No recommendations found for Customer {customer_name}.")

