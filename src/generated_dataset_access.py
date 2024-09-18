import pickle

data = pickle.load(open("../data/generated/atis_gpt-4o_1_200.pkl", "rb"))

print(data)