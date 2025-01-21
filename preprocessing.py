import pandas as pd
import re
from tqdm import tqdm


def preprocess(articles):
    """Preprocess the articles and return a dictionary with the number of digits in each article."""

    pd.set_option('display.max_colwidth', None)

    nos_df = (pd.read_csv(articles,
                        parse_dates=['datetime'], encoding='utf-8') 
            .sort_values("datetime")
            )

    economie_articles = nos_df[nos_df['category'] == 'Economie']
    titles_and_content = economie_articles[['title', 'content']]

    # Make a dictionary with number of digits as key and the article as value
    articles_dict = {}
    for i, row in tqdm(titles_and_content.iterrows(), 
                       total=titles_and_content.shape[0], 
                       desc="Processing articles"):
        content = row['content']

        num_digits = len(re.findall(r'\d', content))
        articles_dict[content] = num_digits

    return articles_dict
