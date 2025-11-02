# === Imports ===
import os
import random
import shutil
import re
from collections import defaultdict
import nltk
import pandas as pd     
import numpy as np
import matplotlib.pyplot as plt
from nltk import word_tokenize, pos_tag, FreqDist
from nltk.corpus import stopwords
from string import punctuation
import contractions
import seaborn as sns

# === NLTK Data ===
nltk.data.path.append('C:/Users/Jennifer/AppData/Roaming/nltk_data')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('universal_tagset')
nltk.download('stopwords')

# === Parameters ===
flat_chunk_dir = "C:/Users/Jennifer/Downloads/essay_chunks"
output_dir = "C:/Users/Jennifer/Documents/balanced_chunk_dataset"
target_range = (100, 200)
chunks_per_author = 6
authors_to_include = 4
stop_words = set(stopwords.words('english'))

# === Author Name Mapping ===
author_name_map = {
    'joan_didion': 'Joan Didion',
    'joandidion': 'Joan Didion',
    'Joan_Didion': 'Joan Didion',
    'davidfosterwallace': 'David Foster Wallace',
    'david_foster_wallace': 'David Foster Wallace',
    'johnjeremiahsullivan': 'John Jeremiah Sullivan',
    'john_jeremiah_sullivan': 'John Jeremiah Sullivan',
    'zadie_smith': 'Zadie Smith',
    'zadiesmith': 'Zadie Smith'
}

# === Reusable Styled Bar Plot Function ===
def plot_bar(df, x_col, y_col, title, ylabel, save_path=None):
    colors = ["#6BBF59", "#A2E4B8", "#457C4F", "#1E4D2B"] * (len(df) // 4 + 1)
    fig, ax = plt.subplots(figsize=(8, 6))
    bars = ax.bar(df[x_col], df[y_col], color=colors[:len(df)])

    for bar in bars:
        height = bar.get_height()
        ax.annotate(f"{height:.2f}", 
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=9)

    ax.set_title(title, fontsize=16, fontweight='bold')
    ax.set_xlabel(x_col, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.tick_params(axis='x', labelrotation=0)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()

# === Core Processing Functions ===
def get_valid_chunks(base_dir, word_range):
    author_chunks = defaultdict(list)
    for root, _, files in os.walk(base_dir):
        for file in files:
            if file.endswith(".txt"):
                path = os.path.join(root, file)
                try:
                    with open(path, encoding="utf-8") as f:
                        content = f.read()
                except UnicodeDecodeError:
                    with open(path, encoding="latin-1") as f:
                        content = f.read()
                wc = len(content.split())
                if word_range[0] <= wc <= word_range[1]:
                    rel_path = os.path.relpath(path, base_dir)
                    author = rel_path.split(os.sep)[0]
                    author_chunks[author].append((path, wc))
    return author_chunks

def sample_chunks(author_chunks, output_dir, per_author, num_authors):
    eligible = [a for a, chunks in author_chunks.items() if len(chunks) >= per_author]
    selected = random.sample(eligible, min(num_authors, len(eligible)))
    os.makedirs(output_dir, exist_ok=True)

    texts = []
    metadata = []

    for author in selected:
        samples = random.sample(author_chunks[author], per_author)
        for i, (src_path, wc) in enumerate(samples):
            filename = f"{author}_{os.path.basename(src_path)}"
            dst_path = os.path.join(output_dir, filename)
            shutil.copy(src_path, dst_path)
            with open(src_path, encoding="utf-8") as f:
                text = f.read()
            texts.append(text)
            metadata.append({"author": author, "chunk_id": i + 1})
            print(f"{author}: ✅ {os.path.basename(src_path)} ({wc} words)")
    return texts, metadata

def pos_analysis(texts, metadata):
    records = []
    for i, text in enumerate(texts):
        tokens = word_tokenize(text)
        tokens = [re.sub(r"[^\w\s]", "", t) for t in tokens if re.sub(r"[^\w\s]", "", t)]
        tags = pos_tag(tokens, tagset="universal")
        for word, tag in tags:
            records.append({
                "author": metadata[i]["author"],
                "chunk_id": metadata[i]["chunk_id"],
                "word": word,
                "pos": tag
            })
    return pd.DataFrame(records)

def clean_tokens(tokens):
    tokens = [contractions.fix(w) for w in tokens]
    return [w for w in tokens if w.lower() not in stop_words and w not in punctuation]

def freq_analysis_from_texts(texts, metadata):
    freq_dists = {}
    unique_counts = {}
    total_counts = {}
    lexical_diversity = {}

    for i, text in enumerate(texts):
        tokens = word_tokenize(text.lower())
        name = f"{metadata[i]['author']}{metadata[i]['chunk_id']}"
        dist = FreqDist(tokens)

        freq_dists[name] = dist
        unique_counts[name] = len(set(tokens))
        total_counts[name] = len(tokens)
        lexical_diversity[name] = (len(set(tokens)) / len(tokens)) * 100

    return freq_dists, unique_counts, total_counts, lexical_diversity

def ave_lexical_diversity(lexical_diversity):
    grouped = defaultdict(list)
    for name, value in lexical_diversity.items():
        author = ''.join(filter(str.isalpha, name))
        grouped[author].append(value)
    return {author.upper(): np.mean(values) for author, values in grouped.items()}

def word_length_analysis(texts, metadata):
    records = []
    for i, text in enumerate(texts):
        tokens = word_tokenize(text)
        tokens = [re.sub(r"[^\w\s]", "", t) for t in tokens if re.sub(r"[^\w\s]", "", t)]
        lengths = [len(token) for token in tokens if token.isalpha()]
        avg_length = sum(lengths) / len(lengths) if lengths else 0
        records.append({
            "author": metadata[i]["author"],
            "chunk_id": metadata[i]["chunk_id"],
            "avg_word_length": avg_length
        })
    return pd.DataFrame(records)

def hapax_legomena_analysis(texts, metadata):
    records = []
    for i, text in enumerate(texts):
        tokens = word_tokenize(text.lower())
        dist = FreqDist(tokens)
        hapaxes = dist.hapaxes()
        num_hapax = len(hapaxes)
        total_tokens = len(tokens)
        ratio = (num_hapax / total_tokens) * 100 if total_tokens > 0 else 0
        records.append({
            "author": metadata[i]["author"],
            "chunk_id": metadata[i]["chunk_id"],
            "hapax_count": num_hapax,
            "hapax_ratio": ratio
        })
    return pd.DataFrame(records)

# === Stylometric Run Block ===
chunks = get_valid_chunks(flat_chunk_dir, target_range)
texts, metadata = sample_chunks(chunks, output_dir, chunks_per_author, authors_to_include)

# Analysis
df = pos_analysis(texts, metadata)
word_len_df = word_length_analysis(texts, metadata)
freq_dists, unique_counts, total_counts, lexical_diversity = freq_analysis_from_texts(texts, metadata)
clean_ave_div_author = ave_lexical_diversity(lexical_diversity)
hapax_df = hapax_legomena_analysis(texts, metadata)

# === Aggregate + Format for Charts ===
# POS tags
pos_counts = df.groupby(['author', 'chunk_id'])['pos'].value_counts().rename('count').reset_index()
avg_pos = pos_counts.groupby(['pos', 'author'])['count'].mean().reset_index(name='average_count')
pivot_pos = avg_pos.pivot(index='author', columns='pos', values='average_count').fillna(0)
mapped_index = pivot_pos.index.str.lower().map(author_name_map)
pivot_pos.index = mapped_index.where(mapped_index.notna(), pivot_pos.index)
pivot_pos = pivot_pos.sort_index()

# Lexical diversity 
lex_div_df = pd.DataFrame([
    {"Author": author_name_map.get(k.lower(), k), "Lexical Diversity": v}
    for k, v in clean_ave_div_author.items()
])
lex_div_df.sort_values("Author", inplace=True)

# Word length
author_avg_word_len = word_len_df.groupby("author")["avg_word_length"].mean().reset_index()
author_avg_word_len["Author"] = author_avg_word_len["author"].str.lower().map(author_name_map).fillna(author_avg_word_len["author"])
author_avg_word_len = author_avg_word_len[["Author", "avg_word_length"]].rename(columns={"avg_word_length": "Average Word Length"})
author_avg_word_len.sort_values("Author", inplace=True)

# Hapax
author_avg_hapax = (
    hapax_df.groupby("author")["hapax_ratio"]
    .mean()
    .reset_index()
    .rename(columns={"author": "Author", "hapax_ratio": "Hapax Ratio (%)"})
)
author_avg_hapax["Author"] = author_avg_hapax["Author"].str.lower().map(author_name_map).fillna(author_avg_hapax["Author"])
author_avg_hapax.sort_values("Author", inplace=True)

# Unique words
grouped_unique = defaultdict(list)
for name, count in unique_counts.items():
    author = re.match(r"[a-zA-Z_]+", name).group().rstrip('_').lower()
    grouped_unique[author].append(count)
unique_word_df = pd.DataFrame({
    "Author": [author_name_map.get(a.lower(), a) for a in grouped_unique],
    "Average Unique Words": [np.mean(c) for c in grouped_unique.values()]
})
unique_word_df.sort_values("Author", inplace=True)

# === Final Styled Plots ===
plot_bar(lex_div_df, "Author", "Lexical Diversity", "Average Lexical Diversity per Author", "Lexical Diversity (%)")
plot_bar(author_avg_word_len, "Author", "Average Word Length", "Average Word Length per Author", "Word Length")
plot_bar(author_avg_hapax, "Author", "Hapax Ratio (%)", "Hapax Legomena Ratio per Author", "Hapax Ratio (%)")
plot_bar(unique_word_df, "Author", "Average Unique Words", "Average Unique Words per Author", "Unique Word Count")

# POS Tags
pivot_pos = pivot_pos.drop(columns=["X", "."], errors="ignore")

ax = pivot_pos.plot(
    kind='bar',
    stacked=True,
    figsize=(12, 6),
    colormap="Greens",
    edgecolor='white',
    linewidth=0.5
)

plt.title("Average POS Tag Distribution per Author", fontsize=16, fontweight="bold")
plt.xlabel("Author")
plt.ylabel("Average Count")
plt.legend(title="POS", bbox_to_anchor=(1.05, 1), loc="upper left")

# Rotate author names
ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')  # or rotation=90 for vertical

# Optional: Add value labels
for container in ax.containers:
    for bar in container:
        height = bar.get_height()
        if height > 0:
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_y() + height / 2,
                f'{height:.1f}',
                ha='center',
                va='center',
                fontsize=8
            )

plt.figure(figsize=(12, 8))

# Create the heatmap
sns.heatmap(
    pivot_pos, 
    annot=True, 
    fmt=".1f", 
    cmap="Greens", 
    linewidths=0.5, 
    linecolor='white',
    cbar_kws={'label': 'Average Count'}
)

# Titles and labels
plt.title("Average POS Tag Distribution per Author", fontsize=16, fontweight="bold")
plt.xlabel("POS Tag", fontsize=12)
plt.ylabel("Author", fontsize=12)

# Rotate x-axis labels for readability
plt.xticks(rotation=45, ha='right')

plt.tight_layout()
plt.show()

