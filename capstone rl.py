# -------------------------------------------------
# SMARTMART AI SUPERMARKET BOT
# Combines NLP (Intent Detection) + Reinforcement Learning (Q-Learning)
# -------------------------------------------------

import numpy as np
import random
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import matplotlib.pyplot as plt


# -------------------------------------------------
# MODULE 2 — NLP MODEL (AI PART)
# -------------------------------------------------

# Training sentences
training_sentences = [
    "hi", "hello",
    "what items available", "show stock", "what do you have",
    "price of rice", "cost of milk", "how much is bread",
    "suggest something", "what should i buy", "recommend",
    "bye", "exit"
]

training_labels = [
    "greeting", "greeting",
    "check_stock", "check_stock", "check_stock",
    "check_price", "check_price", "check_price",
    "recommend", "recommend", "recommend",
    "goodbye", "goodbye"
]

vectorizer = CountVectorizer()
X = vectorizer.fit_transform(training_sentences)

model = MultinomialNB()
model.fit(X, training_labels)

def detect_intent(user_input):
    X_test = vectorizer.transform([user_input])
    return model.predict(X_test)[0]

# -------------------------------------------------
# SUPERMARKET INVENTORY DATABASE
# -------------------------------------------------

inventory = {
    "rice": {"stock": 20, "price": 50},
    "milk": {"stock": 15, "price": 30},
    "bread": {"stock": 8, "price": 25},
    "eggs": {"stock": 30, "price": 6},
    "apple": {"stock": 25, "price": 80}
}

# -------------------------------------------------
# MODULE 3 — REINFORCEMENT LEARNING SETUP
# -------------------------------------------------

states = ["greeting", "check_stock", "check_price", "recommend", "goodbye"]
actions = ["show_items", "show_price", "suggest_items"]

state_index = {s: i for i, s in enumerate(states)}
action_index = {a: i for i, a in enumerate(actions)}

Q = np.zeros((len(states), len(actions)))

alpha = 0.1   # learning rate
gamma = 0.9   # discount
epsilon = 0.3 # exploration

# -------------------------------------------------
# ACTION SELECTION (ε-Greedy)
# -------------------------------------------------

def choose_action(state):
    s = state_index[state]

    if random.uniform(0,1) < epsilon:
        return random.choice(actions)
    else:
        return actions[np.argmax(Q[s])]

# -------------------------------------------------
# BOT RESPONSE ENGINE
# -------------------------------------------------

def perform_action(intent, user_text, action):

    words = user_text.lower().split()

    product = None
    for w in words:
        if w in inventory:
            product = w
            break

    if action == "show_items":
        items = ", ".join(inventory.keys())
        return f"Available items: {items}"

    elif action == "show_price" and product:
        return f"{product} costs ₹{inventory[product]['price']}"

    elif action == "suggest_items":
        suggestions = [p for p in inventory if inventory[p]["stock"] > 10]
        return "Recommended items: " + ", ".join(suggestions)

    return "Please ask about products."

# -------------------------------------------------
# REWARD FUNCTION
# -------------------------------------------------

def get_reward(intent, action):
    if intent == "recommend" and action == "suggest_items":
        return 10
    elif intent == "check_price" and action == "show_price":
        return 8
    elif intent == "check_stock" and action == "show_items":
        return 8
    elif intent == "goodbye":
        return 5
    else:
        return -5

# -------------------------------------------------
# Q-LEARNING UPDATE
# -------------------------------------------------

def update_q(state, action, reward):
    s = state_index[state]
    a = action_index[action]

    Q[s,a] = Q[s,a] + alpha * (reward + gamma * np.max(Q[s]) - Q[s,a])
episode_rewards = []
total_reward = 0
step_count = 0
existing_rewards = []
rl_rewards = []

def existing_method_reward(intent):
    # Rule-based system gives fixed response (no learning)
    if intent in ["check_stock", "check_price", "recommend"]:
        return 5   # average static performance
    return 2

# -------------------------------------------------
# CHAT LOOP (TRAINING THROUGH INTERACTION)
# -------------------------------------------------

print("🛒 SmartMart RL Assistant Ready!")
print("Ask about stock, price, or suggestions.\n")

while True:
    user = input("You: ")

    intent = detect_intent(user)

    action = choose_action(intent)

    response = perform_action(intent, user, action)

    # ---- Existing System Reward ----
    base_reward = existing_method_reward(intent)
    existing_rewards.append(base_reward)

    # ---- RL System Reward ----
    reward = get_reward(intent, action)
    update_q(intent, action, reward)
    rl_rewards.append(reward)



    print("Bot:", response)

    if intent == "goodbye":
        break
# -------------------------------
# PLOT LEARNING PERFORMANCE
# -------------------------------

plt.plot(existing_rewards, label="Multinomial Naïve Bayes", linestyle="--")
plt.plot(rl_rewards, label="Proposed RL-Based", linewidth=2)

plt.title("Performance Comparison")
plt.xlabel("Interactions")
plt.ylabel("Reward")
plt.legend()
plt.show()

