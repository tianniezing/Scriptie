import re
import math
import itertools
import pandas as pd
from tabulate import tabulate
from tqdm import tqdm
from preprocessing import preprocess
from transformers import AutoModelForCausalLM, AutoTokenizer
from perplexity import calculate_perplexity


def secret_to_octal(secret_message):
    """
    Convert a secret message to octal numbers and add end of text character in octal numbers
    """

    octal_list = [oct(ord(char))[2:] for char in secret_message]
    octal_list.append(oct(3)[2:])
    return octal_list

def calculate_mn(octal_list):
    """
    Calculates from a list of octal numbers a list of M,N numberpairs and returns the list
    """

    mn_list = []

    q = 3
    x = 3
    y = 5
    z = 2

    for octal_number in octal_list:
        # Coefficients
        a = int(1)
        b = int(1)
        c = (int(octal_number)*2)*-1

        # Calculate the discriminant
        discriminant = b**2 - 4*a*c

        # Check if discriminant is non-negative (real roots)
        if discriminant >= 0:
            # Calculate both roots
            root1 = (-b + math.sqrt(discriminant)) / (2 * a)
            root2 = (-b - math.sqrt(discriminant)) / (2 * a)
            
            # Select the positive root
            positive_root = root1 if root1 > 0 else root2
            
            # Calculate m
            m = (math.floor(positive_root) * z + (x * (c % q))) % 21
            
            # Calculate n
            n = (int(abs(c/2) - m*((m+1)/2)) - y) % 21
            
            pair = f"{str(m).zfill(2)}{str(n).zfill(2)}"
            mn_list.append(pair)
        else:
            print("The equation has no real solutions.")
    return mn_list

def calculate_mn_table(octal_number):
    """
    Calculates the M,N numberpair from an octal number and returns the M and N values.
    """

    q = 3
    x = 3
    y = 5
    z = 2

    # Coefficients
    a = int(1)
    b = int(1)
    c = (int(octal_number)*2)*-1

    # Calculate the discriminant
    discriminant = b**2 - 4*a*c

    # Check if discriminant is non-negative (real roots)
    if discriminant >= 0:
        # Calculate both roots
        root1 = (-b + math.sqrt(discriminant)) / (2 * a)
        root2 = (-b - math.sqrt(discriminant)) / (2 * a)
        
        # Select the positive root
        positive_root = root1 if root1 > 0 else root2
        
        # Calculate m
        m = (math.floor(positive_root) * z + (x * (c % q))) % 21
        
        # Calculate n
        n = (int(abs(c/2) - m*((m+1)/2)) - y) % 21

        return m, n

def create_mn_table(k=21, l=21):
    """
    Create a table with M, N combinations and their corresponding octal values
    """

    # Create a dictionary to store M, N combinations and their octal values
    mn_table = {M: {N: [] for N in range(l)} for M in range(k)}

    numbers_list = [
        0, 1, 2, 3, 4, 5, 6, 7,
        10, 11, 12, 13, 14, 15, 16, 17,
        20, 21, 22, 23, 24, 25, 26, 27,
        30, 31, 32, 33, 34, 35, 36, 37,
        40, 41, 42, 43, 44, 45, 46, 47,
        50, 51, 52, 53, 54, 55, 56, 57,
        60, 61, 62, 63, 64, 65, 66, 67,
        70, 71, 72, 73, 74, 75, 76, 77,
        100, 101, 102, 103, 104, 105, 106, 107,
        110, 111, 112, 113, 114, 115, 116, 117,
        120, 121, 122, 123, 124, 125, 126, 127,
        130, 131, 132, 133, 134, 135, 136, 137,
        140, 141, 142, 143, 144, 145, 146, 147,
        150, 151, 152, 153, 154, 155, 156, 157,
        160, 161, 162, 163, 164, 165, 166, 167,
        170, 171, 172, 173, 174, 175, 176, 177
    ]

    for octal_value in numbers_list:
        M, N = calculate_mn_table(octal_value)
        mn_table[M][N].append(f"{octal_value}") 

    # Convert to DataFrame for better visualization
    mn_table_df = pd.DataFrame({N: {M: ", ".join(mn_table[M][N]) for M in mn_table} for N in range(l)})
    mn_table_df.index.name = "M \\ N"

    # # Save the table
    mn_table_df.to_csv("M_N_Octal_Mapping_Table_v2.csv")
    print("Table saved as 'M_N_Octal_Mapping_Table_v2.csv'")


def find_articles(mn_list, articles_dict):
    """Find the first 50 shortest articles in the articles_dict that have at least the same number of digits 
    as the M,N number pairs list and don't contain numbers between 2000 and 2050
    """

    num_digits_mn = sum(len(pair) for pair in mn_list)
    matching_articles = []

    for article, num_digits in articles_dict.items():
        if num_digits >= num_digits_mn:
            # Check if the article contains numbers between 2000 and 2050
            if not re.search(r'\b(200[0-9]|201[0-9]|202[0-9]|203[0-9]|204[0-9]|2050)\b', article):
                matching_articles.append(article)
            # matching_articles.append(article)

    matching_articles.sort(key=len)

    return matching_articles[:50]

def modify_articles(article_list, mn_list):
    """
    Modify the articles by replacing digits with the digits from the M,N number pairs
    """

    pairs_str = ''.join(mn_list)
    modified_article_list = []

    for article in tqdm(article_list, desc="Modify articles"):
        pair_index = 0 

        def replace_digit(match):
            nonlocal pair_index
            if pair_index < len(pairs_str):
                replacement = pairs_str[pair_index]
                pair_index += 1
                return replacement
            else:
                # Return the original digit if no more pairs are available
                return match.group()

        modified_article = re.sub(r'\d', replace_digit, article)
        modified_article_list.append((article, modified_article))

    return modified_article_list

def select_best_article(modified_articles):
    """
    Select the best article based on the perplexity of the modified article.
    """

    article_perplexity = {}

    for original_article, modified_article in tqdm(modified_articles, desc="Calculate perplexity per modified article"):
        perplexity = calculate_perplexity(modified_article)
        article_perplexity[perplexity] = (original_article, modified_article)
    
    sorted_article_perplexity = dict(sorted(article_perplexity.items()))

    return dict(itertools.islice(sorted_article_perplexity.items(), 1))

def encode(secret):
    """
    Encode a secret message into an article and return the original article, 
    original article perplexity, modified article, modified article perplexity, and payload capacity.
    """

    octal_list = secret_to_octal(secret)
    mn_list = calculate_mn(octal_list)
    articles_dict = preprocess("dutch-news-articles.csv")
    articles = find_articles(mn_list, articles_dict)
    modified_articles = modify_articles(articles, mn_list)
    best_article = select_best_article(modified_articles)

    best_perplexity = list(best_article.keys())[0]
    original_article, modified_article = best_article[best_perplexity]

    original_article_perplexity = calculate_perplexity(original_article)
    payload_capacity = articles_dict[original_article]/len(original_article)

    print("Met jaartallen")
    print("Octal list:", octal_list)
    print("MN pairs:", mn_list)
    print("Modified article char length:", len(modified_article))
    print("Article number of digits:", articles_dict[original_article])
    print("Modified article payload capacity", articles_dict[original_article]/len(original_article))
    print("Modified article perplexity:", best_perplexity)
    print("Original article:", original_article)
    print("Modified article:", modified_article)

    return original_article, original_article_perplexity, modified_article, best_perplexity, payload_capacity
