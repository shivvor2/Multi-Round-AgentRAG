import nltk
from nltk.tokenize import sent_tokenize

# Downloads the punkt model
nltk.download('punkt')

# WARNING: Assumes "Languages with romanic characters" e.g. English French Spanish etc only, DOES NOT WORK WITH CHINESE/ JAPANESE/ KOREAN etc
# Over-engineering go crazy, the exact token count doesnt matter much anyways because the embedding model can use a different embedding compared to the space anyways
# Returns a list of dictionaries with members: "text", "chunk_length"
def sentence_level_chunking(text, chunk_size = 256, embedding_tokenizer = None, estimate_token_count: bool = False, token_per_word_ratio = 0.75):
    sentences = sent_tokenize(text)
    chunks = []
    current_chunk = []
    current_chunk_length = 0

    # This looks funny because "sentences" is not a list and can only assume iterator properties + exception case to handle long sentences
    for sentence in sentences:
        to_process = [sentence]
        while to_process: # At least 1 member in to_process
            sentence_length = token_count(to_process[0], embedding_tokenizer, estimate_token_count, token_per_word_ratio)
            
            # Chunk not filled up to chunk size limit
            if current_chunk_length + sentence_length <= chunk_size:
                current_chunk.append(to_process[0])
                current_chunk_length += sentence_length
            
            # Chunk reached size limit but current sentence shorter then max chunk size
            elif sentence_length <= chunk_size:
                chunks.append(create_chunk_dict(current_chunk, current_chunk_length))
                current_chunk = []
                current_chunk_length = 0
                to_process.append(sentence) # TODO: Same sentence length is recalculated next iteration, fix it.

            # Chunk will reached size limit & sentence_length >= chunk_size
            else: 
                split_sentences = []
                if estimate_token_count:
                    split_sentences = split_sentences_estimate_tokencount(sentence, chunk_size, token_per_word_ratio)
                else: # estimate_token_count = false
                    split_sentences = split_sentences_no_estimation(sentence, sentence_length, chunk_size, embedding_tokenizer)
                to_process = to_process.extend(split_sentences)
            to_process.pop()

    return chunks

def token_count(sentence, embedding_tokenizer, estimate_token_count: bool, token_per_word_ratio):
    sentence_length = -1
    if estimate_token_count:
        sentence_length = len(sentence.split())*token_per_word_ratio
    else:
        try:
            sentence_length = len(embedding_tokenizer.tokenize(sentence))
        except:
            raise TypeError(f"Embedding_tokenizer is of invalid type: {type(embedding_tokenizer)}") # I don't like this
    return sentence_length

def create_chunk_dict(current_chunk, current_length):
    chunk_dict = {
        "text": " ".join(current_chunk),
        "chunk_length": current_length,
    }
    return chunk_dict

def split_sentences_estimate_tokencount(sentence, chunk_size, token_per_word_ratio):
    split_sentences_words = sentence.split()
    words_per_chunk = int(chunk_size * token_per_word_ratio)
    split_sentences = [split_sentences_words[i:i+words_per_chunk] for i in range(0, len(split_sentences_words), words_per_chunk)]
    split_sentences_string = " ".join(split_sentences)
    return split_sentences_string

# Case for no estimation of token_count is bad (since I don't know how to get the thing to select the first {chunk_size} items)
# For now, we use the same approach as the "estimate tokencount" case, except that we calculate the token per word ratio by using the sentence length obtained from the token_count function
def split_sentences_no_estimation(sentence, sentence_length, chunk_size, embedding_tokenizer):
    split_sentences_words = sentence.split()
    token_per_word_ratio = sentence_length/ len(split_sentences_words)
    words_per_chunk = int(chunk_size * token_per_word_ratio)
    split_sentences = [split_sentences_words[:words_per_chunk], split_sentences_words[words_per_chunk:]]
    split_sentences_string = " ".join(split_sentences)
    return split_sentences_string