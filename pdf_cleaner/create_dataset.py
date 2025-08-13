import os
import pandas as pd
import spacy
import openai
from langchain.text_splitter import RecursiveCharacterTextSplitter
from rake_nltk import Rake
import dotenv
import json

# Load environment variables from .env file
dotenv.load_dotenv()

# Load chapter summaries from the JSON file
with open('bhagavad_gita_summaries.json', 'r') as f:
    chapter_summaries = json.load(f)

# Initialize spaCy for keyword extraction
nlp = spacy.load("en_core_web_sm")

# Define Chapter Titles (replace with actual titles)
chapter_titles = {
    1: "Chapter 1: Arjuna Vishada Yoga, The Yoga of Despondency of Arjuna",
    2: "Chapter 2: Sankhya Yoga, The Yoga of Knowledge ",
    3: "Chapter 3: Karma Yoga, The Yoga of Action",
    4: "Chapter 4: Jnana Karma Sanyasa Yoga, The Yoga of Renunciation of Action with Knowledge",
    5: "Chapter 5: Karma Sanyasa Yoga, The Yoga of Renunciation of Action",
    6: "Chapter 6: Dhyana Yoga, The Yoga of Meditation",
    7: "Chapter 7: Jnana Vijnana Yoga, The Yoga of Wisdom and knowledge",
    8: "Chapter 8: Akshara Brahma Yoga, The Yoga of Imperishable Brahman",
    9: "Chapter 9: Raja Vidya Raja Guhya Yoga (The Yoga of Royal Knowledge and Royal Secret)",
    10: "Chapter 10: Vibhuti Yoga (The Yoga of Divine Glories)",
    11: "Chapter 11: Visvarupa Darshana Yoga (The Yoga of the Vision of the Universal Form)",
    12: "Chapter 12: Bhakti Yoga (The Yoga of Devotion)",
    13: "Chapter 13: Kshetra Kshetragna Vibhaga Yoga (The Yoga of the Field and the Knower of the Field)",
    14: "Chapter 14: Gunatraya Vibhaga Yoga (The Yoga of the Division of the Three Gunas)",
    15: "Chapter 15: Purusottama Yoga (The Yoga of the Supreme Person)",
    16: "Chapter 16: Daivasura Sampad Vibhaga Yoga (The Yoga of the Division between the Divine and the Non-Divine)",
    17: "Chapter 17: Sraddhatraya Vibhaga Yoga (The Yoga of the Threefold Faith)",
    18: "Chapter 18: Moksha Sanyasa Yoga (The Yoga of Liberation and Renunciation)"
}

# Initialize the text splitter
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=700,
    chunk_overlap=20,
    length_function=len,
    is_separator_regex=False,
)


# Function to extract keywords using spaCy
def extract_keywords(text):
    doc = nlp(text)
    keywords = [chunk.text for chunk in doc.noun_chunks]
    return ", ".join(set(keywords))


# Function to process each chapter file
def process_chapter(chapter_number, chapter_file):
    with open(chapter_file, 'r', encoding='utf-8') as file:
        text = file.read()

    # Split the text into chunks
    chunks = text_splitter.create_documents([text])

    # Get the chapter title and short description from the JSON data
    chapter_title = chapter_titles.get(chapter_number, f"Chapter {chapter_number}")
    short_description = chapter_summaries.get(f"Chapter {chapter_number}", {}).get('summary', "No summary available.")

    # Append each chunk along with metadata to the data list
    for chunk in chunks:
        keywords = extract_keywords(chunk.page_content)
        data.append({
            'Chapter Number': chapter_number,
            'Chapter Title': chapter_title,
            'Short Description': short_description,
            'Keywords': keywords,
            'Chunk Content': chunk.page_content
        })


# Define the path to your chapter text files
chapter_folder = './chapters'  # Adjust this path
output_csv = 'chapter_data.csv'

# Initialize an empty list to store metadata and chunk content
data = []

# Iterate over each chapter file in the folder
for i in range(1, 19):  # Assuming you have 18 chapters
    chapter_file = os.path.join(chapter_folder, f'Chapter-{i}.txt')  # Ensure chapter filenames are correctly named
    if os.path.exists(chapter_file):
        process_chapter(i, chapter_file)
    else:
        print(f"Chapter file {chapter_file} not found!")

# Convert the data list into a pandas DataFrame
df = pd.DataFrame(data)

# Save the DataFrame to CSV
df.to_csv(output_csv, index=False)

print(f"CSV file created successfully: {output_csv}")
