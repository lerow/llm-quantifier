
import torch
import re
import matplotlib.pyplot as plt
import numpy as np

import matplotlib, json


quantifiers = {
     "less than half": lambda total, num1, num2: num1 < total / 2
}

'''
quantifiers = {
    "at least 3": lambda total, num1, num2: num1 >= 3,
    "at least 4": lambda total, num1, num2: num1 >= 4,
    "at most 5": lambda total, num1, num2: num1 <= 5,
    "at most 6": lambda total, num1, num2: num1 <= 6,
    "more than 1": lambda total, num1, num2: num1 > 1,
    "more than 5": lambda total, num1, num2: num1 > 5,
    "more than 10": lambda total, num1, num2: num1 > 10,
    "all": lambda total, num1, num2: num1 == total,
    "none": lambda total, num1, num2: num1 == 0,
    "between 4 and 6": lambda total, num1, num2: 4 <= num1 <= 6,
    "between 2 and 10": lambda total, num1, num2: 2 <= num1 <= 10,
    "at most half": lambda total, num1, num2: num1 <= total / 2,
    "more than half": lambda total, num1, num2: num1 > total / 2,
    "less than half": lambda total, num1, num2: num1 < total / 2,
    "at least half": lambda total, num1, num2: num1 >= total / 2,
} '''



def remove_non_letter(s):
    return re.sub(r'[^a-zA-Z]', '', s)


def parse_output(s: str) -> int:
    # if true return 1 else return 0

    s = s.strip().split()

    for token in s:
        if len(token) < 7:
            token = remove_non_letter(token).lower()
            if token == "true":
                return 1
            if token == "false":
                return 0

    return 0

# common, less common,
# rare, rarer,
# rarest
def generate_freq_data():
    f = open("./data/madeup_at_least_half.txt", "w")

    common = ["books", "chairs", "doors",
    "participants", "activities", "systems",
    "wars", "blocks", "words", "reports"]

    less_common = ["crowds", "negotiations", "cup holders",
    "arteries", "identifiers", "payrolls",
    "hostages", "coupons", "remedies", "butterflies"]

    rare = ["jaws", "turbines", "rooftops",
    "hikers", "purses", "empires", "insurers",
    "camels", "entitlements", "coils"]

    rarer = ["auroras", "borrowers", "fasteners",
    "headscarves", "hickories", "geneticists",
    "catapults", "blurbs", "glaciers", "eyewitnesses"]

    rarest = ["ocean basins", "jests", "lidars",
    "inequalities", "microchips",
    "humanoids", "philanthropies",
    "medullas", "ornamentals", "jabs"]

    madeup = ["blexes", "ptexes", "rangaloons", "Treslings", "baroue",
              "questries", "sarphines", "zofonikl", "arijkes", "bnouba"]

    objects = madeup
    count = 0
    tr_count = 0

    # less than half
    for obj in objects:

        for num in range(0, 51):
            for q in quantifiers:
                copula1 = "is" if num == 49 else "are"
                copula2 = "is" if num == 1 else "are"

                prompt = "There are 50 " + obj + ". "
                prompt = prompt + str(50 - num) + " of the " + obj + " " + copula1 + " large. "
                prompt = prompt + str(num) + " of the " + obj + " " + copula2 + " small. "

                prompt = prompt + "Are " + q + " of the " + obj + " small? Answer with only one word, true or false."

                gold_truth = quantifiers[q](50, num, 50 - num)

                line = {"input": prompt, "target": str(gold_truth).lower()}

                f.write(str(line).replace("\'", "\""))
                f.write("\n")


    for obj in objects:

        for num in range(0, 51):
            for q in quantifiers:
                copula1 = "is" if num == 49 else "are"
                copula2 = "is" if num == 1 else "are"

                prompt = "There are 50 " + obj + ". "
                prompt = prompt + str(50 - num) + " of the " + obj + " " + copula1 + " large. "
                prompt = prompt + str(num) + " of the " + obj + " " + copula2 + " small. "

                prompt = prompt + "Are " + q + " of the " + obj + " large? Answer with only one word, true or false."

                gold_truth = quantifiers[q](50, 50 - num, num)

                line = {"input": prompt, "target": str(gold_truth).lower()}

                f.write(str(line).replace("\'", "\""))
                f.write("\n")


    f.close()


def generate_data():
    f = open("./data/less_than_half_NEW.txt", "w")

    objects = ["tables", "chairs", "circles", "squares", "apples", "bikes", "pans",
               "trees", "shelves", "birds", "penguins", "mountains"]

    # less than half
    for obj in objects:

        for num in range(0, 51):
            for q in quantifiers:
                copula1 = "is" if num == 49 else "are"
                copula2 = "is" if num == 1 else "are"

                prompt = "There are 50 " + obj + ". "
                prompt = prompt + str(50 - num) + " of the " + obj + " " + copula1 + " large. "
                prompt = prompt + str(num) + " of the " + obj + " " + copula2 + " small. "

                prompt = prompt + "Are " + q + " of the " + obj + " small? Answer with only one word, true or false."

                gold_truth = quantifiers[q](50, num, 50 - num)

                line = {"input": prompt, "target": str(gold_truth).lower()}

                f.write(str(line).replace("\'", "\""))
                f.write("\n")



    for obj in objects:

        for num in range(0, 51):
            for q in quantifiers:
                copula1 = "is" if num == 49 else "are"
                copula2 = "is" if num == 1 else "are"

                prompt = "There are 50 " + obj + ". "
                prompt = prompt + str(50 - num) + " of the " + obj + " " + copula1 + " large. "
                prompt = prompt + str(num) + " of the " + obj + " " + copula2 + " small. "

                prompt = prompt + "Are " + q + " of the " + obj + " large? Answer with only one word, true or false."

                gold_truth = quantifiers[q](50, 50 - num, num)

                line = {"input": prompt, "target": str(gold_truth).lower()}

                f.write(str(line).replace("\'", "\""))
                f.write("\n")


    f.close()


def get_data_stats():
    data_file = open("./data/test_data_new.txt", "r")
    lines = data_file.readlines()
    data_file.close()

    t_count = 0
    f_count = 0
    token_count = 0
    for line in lines:
        line = json.loads(line)
        if line["target"] == "true":
            t_count += 1
        else:
            f_count += 1

        token_count += len(line["input"].split())



    print(t_count)
    print(f_count)
    print(token_count / 18360.0)


def plot():
    # Data
    # f1 = [0.684, 0.652, 0.674, 0.669, 0.649]
    acc = [0.554, 0.517, 0.511, 0.508, 0.512]
    x = [0.1, 0.2, 0.3, 0.4, 0.5]

    # Create the plot
    plt.figure(figsize=(14, 16))

    # Plot f1 values with blue color, dot markers and a line
    # plt.plot(x, f1, 'bo-', label='f1', markersize=10)

    # Plot acc values with red color, square markers and a line
    plt.plot(x, acc, color='steelblue', marker='o', label='acc', markersize=10)


    # Set the axis limits
    # plt.xlim(1, 5)
    # plt.ylim(0.4, 0.7)
    x_labels = ["common", "less common", "rare", "rarer", "rarest"]
    plt.xticks(x, x_labels, fontsize=16)
    # plt.xticks(np.arange(0.08, 0.6, step=0.1))
    plt.yticks(np.arange(0.48, 0.6, step=0.005), fontsize=16)

    # Add title and labels
    plt.title("Mistral-7B Accuracy with q=\'at least half\' ", fontsize=22)
    plt.xlabel("Word Frequency", fontsize=18)
    plt.ylabel("Accuracy", fontsize=18)
    plt.legend()

    # Show the plot
    plt.grid(True)
    plt.show()


# plot()
generate_data()
# get_data_stats()
