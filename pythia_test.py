from transformers import GPTNeoXForCausalLM, AutoTokenizer, AutoModelForCausalLM
import torch
import re
import json
import time


device = "cuda:0" if torch.cuda.is_available() else "cpu"
print(device)


# GPTNeoXForCausalLM
model = AutoModelForCausalLM.from_pretrained(
    "EleutherAI/pythia-2.8b",
    revision="step143000",
    cache_dir="./pythia-2.8b/step143000", load_in_8bit = True, device_map = 0
)
# load_in_8bit = True,
# device_map = 0

tokenizer = AutoTokenizer.from_pretrained(
    "EleutherAI/pythia-2.8b",
    revision="step143000",
    cache_dir="./pythia-2.8b/step143000",
)

# model.to(device)




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


def iterate_quant():

    count = 0

    tp, tn, fp, fn = 0, 0, 0, 0

    force_words = ["true", "false"]
    force_ids = [tokenizer(force_words).input_ids]

    f = open("./data/quantifier-generalization/at_least_half_NEW.txt", "r")

    start_time = time.time()
    for line in f.readlines():
        count += 1

        line = json.loads(line)

        prompt = line["input"]
        gold_truth = 1 if line["target"] == "true" else 0

        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        start_index = int(inputs['input_ids'][0].size()[0])
        # do not output prompt

        tokens = model.generate(**inputs, max_new_tokens=20, pad_token_id=tokenizer.eos_token_id,
                                force_words_ids=force_ids, num_beams=4)

        # force_words_ids=force_ids, num_beams = 4

        llm_response = tokenizer.decode(tokens[0][start_index:])

        ans = parse_output(llm_response)  # 1 if true else 0
        if count % 200 == 0:
            if count < 500:
                print("time: " + str(time.time() - start_time))
            print(count)
            print(prompt)
            print(llm_response)
            print(ans)
            print("gold: " + str(gold_truth) + "\n")


        if ans == gold_truth:
            if gold_truth == 1:
                tp += 1
            else:
                tn += 1
        else:
            if gold_truth == 1:
                fp += 1
            else:
                fn += 1


    ff = open("./data/2_8b-new-at-least-half-RESULTS.txt", "w")
    ff.write(str(tp) + "\n")
    ff.write(str(tn) + "\n")
    ff.write(str(fp) + "\n")
    ff.write(str(fn) + "\n")
    ff.close()



    pr = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1 = (2 * pr * recall) / (pr + recall)
    acc = (tp + tn) / (tp + tn + fp + fn)

    print(tp, tn, fp, fn)
    print("precision: " + str(pr))
    print("recall: " + str(recall))
    print("f1: " + str(f1))
    print("acc: " + str(acc))

    f.close()


iterate_quant()

