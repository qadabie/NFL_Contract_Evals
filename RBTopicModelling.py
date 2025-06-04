from textblob import TextBlob
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import nltk
from collections import Counter
from bertopic import BERTopic
from bertopic.vectorizers import ClassTfidfTransformer
from bertopic.dimensionality import BaseDimensionalityReduction
from umap import UMAP
import os
import pickle
import pyarrow as pa

# Download NLTK data
nltk.download('punkt')
nltk.download('stopwords')
pd.set_option('display.max_columns', None)

# Load the CSV file
data = pd.read_csv('RB_data.csv')
data.dropna(subset=['Analysis', 'Strengths', 'Weaknesses'], inplace=True)

# Tokenize and preprocess the text
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    sentences = text.lower().replace('\n', ' ').split('.')
    processed_sentences = []
    
    for sentence in sentences:
        tokens = word_tokenize(sentence)
        tokens = [word for word in tokens if word.isalnum() and word not in stop_words]
        #words_to_remove = ['runner','running','back','could','but','a','in','the','nfl','contact','speed','runs','he','is','get']
        #tokens = [word for word in tokens if word not in words_to_remove]
        
        # Generate bigrams
        bigrams = [''.join(pair) for pair in zip(tokens[:-1], tokens[1:])]
        words_to_remove = ['runner','running','back','could','but','a','in','the','nfl','contact','speed','runs','he','is','get','contact','tackles','burst']
        tokens = [word for word in tokens if word not in words_to_remove]
        #tokens = [word for word in tokens if word.isalnum() and word not in stop_words]

        # Combine tokens and bigrams
        processed_sentences.append(' '.join(bigrams + tokens))
    
    return '. '.join(processed_sentences)

# Combine all text columns into a single column for processing
data['combined_text'] = data[['Analysis', 'Strengths', 'Weaknesses']].fillna('').agg(' '.join, axis=1)

# Preprocess the combined text
data['processed_combined_text']= data['combined_text'].apply(preprocess_text)
# Count the total number of sentences in the combined text
total_sentences = data['combined_text'].apply(lambda text: len(text.split('.'))).sum()
#print(f"Total number of sentences: {total_sentences}")
# Define seed topics
seed_topics = [
    ["topend", "homerun", "breakaway", "quick", "caught", "straightline", "longspeed", "acceleration", "wheels", "outrun", "speed"],
    ["agility","changedirection", "lateralquickness", "stopstart", "footwork", "hips", "twitchy", "suddenness", "shifty", "elusive", "juke"],
    ["vision", "anticipates", "reads", "lanes", "instincts", "decision-making", "processing", "IQ", "patient", "recognizes"],
    ["burst", "throughhole", "explosive", "firststep", "acceleration", "quick", "sudden", "pops", "edge", "explosiveness", "quickness"],
    ["contact","absorbs", "upright", "tackles", "bounces", "churning", "armtackles", "lowcenter", "balance", "extrayards", "bringdown"],
    ["hands", "catcher", "routes", "reliablehands", "catches", "routerunning", "mismatch", "pass-catching", "third-down"],
    ["blitz", "blocks", "anchors", "protection", "willingblocker", "passpro", "picksup", "blockswith", "blitzers", "protectionskills"],
    ["fumble", "fumbles", "putsground", "protectsfootball", "looseball", "secure", "risky", "turnover", "security", "ballsecurity"]
]

# Extract sentences and ensure they're not empty
sentences = []
for text in data['processed_combined_text']:
    for sentence in text.split('.'):
        sentence = sentence.strip()
        if sentence and len(sentence.split()) >= 14:  # Check if sentence is not empty and has at least 10 words
            sentences.append(sentence)

#print(sentences[:5])
# Save the sentences as a 1-column CSV for further analysis
#sentences_df = pd.DataFrame(sentences, columns=['Sentence'])
#sentences_df.to_csv('rb_sentences.csv', index=False)
#print(f"Saved {len(sentences)} sentences to rb_sentences.csv")
# Check if a pre-trained BERTopic model exists
model_path = 'RB_bertopic_model.pkl'
if os.path.exists(model_path):
    with open(model_path, 'rb') as file:
        topic_model = pickle.load(file)
    print("Loaded existing BERTopic model.")
else:
    # Initialize BERTopic with seed word filtering
    empty_dimensionality_reduction = BaseDimensionalityReduction()
    topic_model = BERTopic(
        vectorizer_model=CountVectorizer(),
        seed_topic_list=seed_topics,
        language='english',
        umap_model = UMAP(n_neighbors=2),  # Use empty dimensionality reduction to avoid UMAP
        nr_topics=7,  # Allow the model to determine the optimal number of topics
        calculate_probabilities=True,  # Enable probabilities for better clustering
        ctfidf_model= ClassTfidfTransformer(seed_words=seed_topics,seed_multiplier=1.3,reduce_frequent_words=True)  # Set the minimum percentage of documents a topic must apply to 20%
    )

    # Fit the BERTopic model on the individual sentences
    topic_model = topic_model.fit(sentences)
    # Print the top 10 most common terms for each topic
    topics = topic_model.get_topics()
    print("\nTop 10 terms for each topic:")
    for topic_id, terms in topics.items():
        if topic_id != -1:  # Skip the outlier topic
            # Sort terms by their relevance score (terms are tuples of (word, score))
            top_terms = sorted(terms, key=lambda x: x[1], reverse=True)[:10]
            term_str = ", ".join([f"{term[0]} ({term[1]:.3f})" for term in top_terms])
            print(f"Topic {topic_id}: {term_str}")

    # Save the trained model for future use
    with open(model_path, 'wb') as file:
        pickle.dump(topic_model, file)
    print("Trained and saved new BERTopic model.")

#topic_model.visualize_topics()
# Define a function to process each row
def process_sentences(text):
    topic_counts = Counter()
    topic_sentiments = {topic: 0.0 for topic in range(len(topic_model.get_topics()))}

    for sentence in text.split('.'):
        if sentence.strip():
            # Identify the topic for the sentence
            preprocessed_sentence = preprocess_text(sentence)
            sentiment = TextBlob(sentence).sentiment.polarity
            topics, prob = topic_model.transform([preprocessed_sentence])
            if topics[0] != -1:
                topic_counts[topics[0]] += 1 * prob[0][0]
                topic_sentiments[topics[0]] += sentiment * prob[0][0]
            
            if len(topics) > 1:
                if topics[1] != -1:
                    topic_counts[topics[1]] += 1 * prob[0][1]
                    topic_sentiments[topics[1]] += sentiment * prob[0][1]

            # Calculate the sentiment of the sentence


    # Calculate average sentiment for each topic
    avg_sentiments = {}
    for topic in topic_sentiments:
        if topic_counts[topic] > 0:
            avg_sentiments[topic] = topic_sentiments[topic] / topic_counts[topic]
        else:
            avg_sentiments[topic] = 0
    # Normalize topic counts
    total_count = sum(topic_counts.values())
    if total_count > 0:
        normalized_counts = {topic: count / total_count for topic, count in topic_counts.items()}
    else:
        normalized_counts = topic_counts
    return normalized_counts, avg_sentiments

# Apply the function to the processed_combined_text column
data['processed_combined_text'] = pa.array(data['processed_combined_text']).cast(pa.string())
results = data['processed_combined_text'].apply(process_sentences)
#print(results[0:5])
# Create separate columns for each topic's count and sentiment score
topic_names = {topic_id: topic_model.get_topic(topic_id)[0][0] for topic_id in range(len(topic_model.get_topics())) if topic_model.get_topic(topic_id)}
for topic_id, topic_name in topic_names.items():
    data[f"{topic_name}_count"] = results.map(lambda x: x[0].get(topic_id, 0))
    data[f"{topic_name}_score"] = results.map(lambda x: x[1].get(topic_id, 0))

# Print the head of the dataframe
# Print the distribution of topics in the training sentences
# topic_distribution = topic_model.get_topic_info()
# print("Topic distribution in training sentences:")
# print(topic_distribution[["Topic", "Name", "Count"]])
# print("\nTotal sentences analyzed:", len(sentences))

# Print descriptive statistics for the topic counts in the dataframe
# print("\nTopic count statistics in the processed data:")
# topic_count_cols = [col for col in data.columns if col.endswith('_count')]
# print(data[topic_count_cols].describe())
# # Save the processed data to a new CSV file
data.to_csv('processed_RB_data.csv', index=False)
# Need to instead get the approximate idea of how much that idea is talked about 
# Use the probabilities predicted of each to 
# Print the topic distribution of the trained model
print("\nApproximate Topic Distribution in the BERTopic Model:")
topic_info = topic_model.get_topic_info()
print(topic_info[["Topic", "Name", "Count", "Representation"]])

# Visualize the topic distribution
try:
    distribution_fig = topic_model.visualize_topics()
    print("Topic distribution visualization created successfully.")
    # You can save this visualization if needed
    # distribution_fig.write_html("topic_distribution.html")
except Exception as e:
    print(f"Could not visualize topics due to: {e}")

# Calculate and print the overall percentage of each topic
total_docs = topic_info["Count"].sum()
print("\nOverall Topic Percentages:")
for topic_id, topic_name in topic_names.items():
    if topic_id in topic_info["Topic"].values:
        topic_count = topic_info.loc[topic_info["Topic"] == topic_id, "Count"].values[0]
        percentage = (topic_count / total_docs) * 100
        print(f"Topic {topic_id} ({topic_name}): {percentage:.2f}%")