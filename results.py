import os
import matplotlib.pyplot as plt
from encoding import encode
from encode_with_ai import ai_encoded
from perplexity import calculate_perplexity
from tabulate import tabulate


# Disable parallelism in tokenizers to avoid deadlocks
os.environ["TOKENIZERS_PARALLELISM"] = "false"

def performance_on_input():
    """
    Compare the performance of the corpus-based method and the generative AI method on different secret messages.
    """

    methods = [
        "Modifying article (5 character secret message)",
        "Modifying article (10 character secret message)",
        "Modifying article (15 character secret message)",
        "AI generated article (5 character secret message)",
        "AI generated article (10 character secret message)",
        "AI generated article (15 character secret message)"
    ]

    messages = [
        "Geluk",
        "Fietspad",
        "Samenwerking"
    ]

    results = []

    for _ in range(1):
        for message in messages:
            # AI generated articles
            ai_encoded_article, _ = ai_encoded(message, 'gpt-4o-mini', 0.6, max_attempts=15)
            perplexity_ai_article = calculate_perplexity(ai_encoded_article)
            payload_capacity_ai_article = len(message) / len(ai_encoded_article)
            results.append(("Generative AI", message, "", perplexity_ai_article, payload_capacity_ai_article, "", ai_encoded_article))

            # Modified articles
            original_article, original_article_perplexity, modified_article, perplexity_modified_article, payload_capacity_modified_article = encode(message)
            results.append(("Corpus based", message, original_article_perplexity , perplexity_modified_article, payload_capacity_modified_article, original_article, modified_article))

    # Print the results in a table
    table = [["Method", "Message", "Perplexity Original", "Perplexity", "Payload Capacity"]]
    for method, message, original_article_perplexity, perplexity, payload_capacity, original_article, modified_ai_article in results:
        table.append([method, message, original_article_perplexity, perplexity, payload_capacity, original_article, modified_ai_article])
    print(tabulate(table, headers="firstrow", tablefmt="grid"))

    # Plot the scatterplot
    plt.figure(figsize=(10, 5))
    for method, message, _, perplexity, payload_capacity, _, _ in results:
        if method == "Generative AI":
            plt.scatter(payload_capacity, perplexity, label=f'{method} ({message})', marker='o')
        else:
            plt.scatter(payload_capacity, perplexity, label=f'{method} ({message})', marker='x')

    plt.xlabel('Payload Capacity')
    plt.ylabel('Perplexity')
    plt.title('Perplexity vs. Payload Capacity per Message')
    plt.legend()
    plt.tight_layout()
    plt.show()

performance_on_input()