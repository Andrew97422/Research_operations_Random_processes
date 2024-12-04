from collections import defaultdict
from random import random


def remove_punctuation(text):
    return (text.replace('\n', ' ').replace('\t', ' ').replace(',', '').replace('.', '').replace('!', '')
            .replace('?', '').replace(';', '').replace(':', '').replace('(', '').replace(')', '').replace('"', ' '))


def build_frequency_dict(words):
    frequency_dict = defaultdict(lambda: defaultdict(int))
    for word in words:
        frequency_dict[word.lower()][word] += 1
    return frequency_dict


def simple_lemmatizer(words, frequency_dict):
    lemmatized = []
    for word in words:
        word_lower = word.lower()
        if word_lower in frequency_dict:
            most_frequent_form = max(frequency_dict[word_lower], key=frequency_dict[word_lower].get)
            lemmatized.append(most_frequent_form.lower())
        else:
            lemmatized.append(word_lower)
    return lemmatized


def build_markov_chain(text, n_gram):
    text = remove_punctuation(text)
    words = text.split()

    frequency_dict = build_frequency_dict(words)
    words = simple_lemmatizer(words, frequency_dict)

    markov_chain = {}
    for i in range(len(words) - n_gram):
        current_state = tuple(words[i:i + n_gram])
        next_state = words[i + n_gram]

        if current_state not in markov_chain:
            markov_chain[current_state] = {}

        if next_state not in markov_chain[current_state]:
            markov_chain[current_state][next_state] = 0

        markov_chain[current_state][next_state] += 1

    for current_state in markov_chain:
        total = sum(markov_chain[current_state].values())
        for state in markov_chain[current_state]:
            markov_chain[current_state][state] /= total

    return markov_chain


def get_suggestion(chain, words, num_words_to_predict):
    results = []
    current_words = list(words)

    # Генерируем заданное количество слов
    for _ in range(num_words_to_predict):
        next_word = get_next_word(chain, current_words)
        if next_word:
            results.append(next_word)
            current_words.append(next_word)
            current_words.pop(0)  # Поддерживаем падение состояния для n-грамм
        else:
            break
    return ' '.join(results)


def get_next_word(chain, words):
    n = len(words)
    while n > 0:
        current_state = tuple(words[:n])
        if current_state in chain:
            possible_states = chain[current_state]
            cumulative = 0
            r = random()
            for state, weight in possible_states.items():
                cumulative += weight
                if r < cumulative:
                    return state
        else:
            n -= 1
    return None


def main(file_path, input_words, num_words_to_predict):
    with open(file_path, 'r', encoding='utf-8') as file:
        text = file.read()
    n_gram = len(input_words)  # точный результат
    markov_chain = build_markov_chain(text, n_gram)
    suggestion = get_suggestion(markov_chain, simple_lemmatizer(input_words, build_frequency_dict(input_words)),
                                num_words_to_predict)
    return suggestion


file_path = 'science.txt'
input_words = 'req.txt'
with open('req.txt', 'r', encoding='utf-8') as file:
    input_words = file.read()

input_words = remove_punctuation(input_words)
input_words = input_words.split()

num_words_to_predict = int(input('Введите количество слов для предсказания:'))

suggestion = main(file_path, input_words, num_words_to_predict)
print("Решение:", suggestion)
