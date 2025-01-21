import openai
import re
import matplotlib.pyplot as plt
import time
from encoding import secret_to_octal, calculate_mn
from perplexity import calculate_perplexity

# Set your OpenAI API key
openai.api_key = ""

def generate_economic_article(mn_list, chatmodel, temp):
    """
    Generate an economic article where the numbers from mn_list are correctly processed.
    """

    system_content = """
    Je bent een assistent die helpt bij het genereren van geloofwaardige economische artikelen. Het artikel moet:
    - Realistisch en natuurlijk aanvoelen.
    - Relevante economische thema's behandelen, zoals investeringen, kosten, subsidies, en overheidsbeleid.
    - ALLEEN de opgegeven lijst van cijfers gebruiken totdat elke cijfer uit de lijst is verwerkt, in de opgegeven volgorde.
    - Extra getallen mogen pas worden toegevoegd nadat alle cijfers uit de lijst zijn gebruikt.
    - Cijfers logisch integreren in de context, bijvoorbeeld als statistieken of financiële gegevens.
    - Een samenhangend artikel genereren in een formele en objectieve toon.
    - Gebruik geen titel.
    - Gebruik geen witregels, zet alle tekst in 1 paragraaf.
    """

    user_content = f"""
    Gebruik de volgende lijst van cijfers, in de gegeven volgorde:
    {mn_list}

    1. Gebruik ALLEEN deze cijfers totdat ze allemaal één keer zijn verwerkt.
    2. Voeg pas extra getallen toe na verwerking van alle cijfers uit de lijst.
    """

    response = openai.ChatCompletion.create(
        model=chatmodel, 
        messages=[
            {"role": "developer", "content": system_content},
            {"role": "user", "content": user_content}
        ],
        max_tokens=2048,
        temperature=temp
    )

    return response["choices"][0]["message"]["content"].strip()

def validate_generated_article(article, mn_list):
    """
    Check if the article contains all the numbers from the mn_list in the correct order.
    """
    digits_in_modified = re.findall(r'\d', article)

    pairs = []
    for i in range(0, len(digits_in_modified), 4):
        if i + 3 < len(digits_in_modified):
            pair = f"{digits_in_modified[i]}{digits_in_modified[i+1]}{digits_in_modified[i+2]}{digits_in_modified[i+3]}"
            pairs.append(pair)

    return pairs == mn_list

def generate_and_validate_article(mn_list, chatmodel, temp, max_attempts=15):
    """
    Generate and validate an article. Retry if the article does not meet the rules.
    """
    attempts = 0

    while attempts < max_attempts:
        attempts += 1
        print(f"Genereren van artikel, poging {attempts}...")
        article = generate_economic_article(mn_list, chatmodel, temp)

        if validate_generated_article(article, mn_list):
            return article, attempts

    raise ValueError("Maximaal aantal pogingen bereikt. Artikel voldoet niet aan de regels.")

def ai_encoded(secret_message, chatmodel, temp, max_attempts=20):
    mn_list = calculate_mn(secret_to_octal(secret_message))
    try:
        validated_article, attempts = generate_and_validate_article(mn_list, chatmodel, temp, max_attempts)
        return validated_article, attempts
    except ValueError as e:
        return str(e), max_attempts

def compare_models_temperature(secret_message):
    """ 
    Compare the performance of different models and temperatures 
    """
    start_time = time.time() 

    models = ["gpt-4o", "gpt-4o-mini"]
    temperatures = [0.3, 0.6, 0.9]
    results = {model: {temp: [] for temp in temperatures} for model in models}
    attempts_results = {model: {temp: [] for temp in temperatures} for model in models}
    payload_capacity_results = {model: {temp: [] for temp in temperatures} for model in models}

    for model in models:
        for temp in temperatures:
            for _ in range(10):
                ai_encoded_article, attempts = ai_encoded(secret_message, model, temp)
                perplexity = calculate_perplexity(ai_encoded_article)
                digits_in_generated = re.findall(r'\d', ai_encoded_article)
                payload_capacity = len(digits_in_generated) / len(ai_encoded_article)
                results[model][temp].append(perplexity)
                attempts_results[model][temp].append(attempts)
                payload_capacity_results[model][temp].append(payload_capacity)
                print(f'Model: {model}, Temperature: {temp}, Perplexity: {perplexity}, Attempts: {attempts}, Payload Capacity: {payload_capacity}')

    # Plot Perplexity
    plt.figure(figsize=(10, 5))
    data = [results[model][temp] for model in models for temp in temperatures]
    labels = [f'{model} (Temp: {temp})' for model in models for temp in temperatures]
    plt.boxplot(data, labels=labels)
    plt.xlabel('Model and Temperature')
    plt.ylabel('Perplexity')
    plt.title('Perplexity per Model and Temperature')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

    # Plot Attempts
    plt.figure(figsize=(10, 5))
    data = [attempts_results[model][temp] for model in models for temp in temperatures]
    labels = [f'{model} (Temp: {temp})' for model in models for temp in temperatures]
    plt.boxplot(data, labels=labels)
    plt.xlabel('Model and Temperature')
    plt.ylabel('Attempts')
    plt.title('Attempts per Model and Temperature')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

    # Plot Payload Capacity
    plt.figure(figsize=(10, 5))
    data = [payload_capacity_results[model][temp] for model in models for temp in temperatures]
    labels = [f'{model} (Temp: {temp})' for model in models for temp in temperatures]
    plt.boxplot(data, labels=labels)
    plt.xlabel('Model and Temperature')
    plt.ylabel('Payload Capacity')
    plt.title('Payload Capacity per Model and Temperature')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

    # Print averages
    for model in models:
        for temp in temperatures:
            avg_perplexity = sum(results[model][temp]) / len(results[model][temp])
            avg_attempts = sum(attempts_results[model][temp]) / len(attempts_results[model][temp])
            avg_payload_capacity = sum(payload_capacity_results[model][temp]) / len(payload_capacity_results[model][temp])
            print(f'Average for Model: {model}, Temperature: {temp} -> Perplexity: {avg_perplexity}, Attempts: {avg_attempts}, Payload Capacity: {avg_payload_capacity}')

    end_time = time.time() 
    print(f"Total execution time: {end_time - start_time} seconds")
