import ssl
import random
import warnings
import re

import nltk
import spacy
from nltk.tokenize import word_tokenize
from nltk.corpus import wordnet
from sentence_transformers import SentenceTransformer, util
from textblob import TextBlob

warnings.filterwarnings("ignore", category=FutureWarning)

NLP_GLOBAL = spacy.load("en_core_web_sm")

def download_nltk_resources():
    """
    Download required NLTK resources if not already installed.
    """
    try:
        _create_unverified_https_context = ssl._create_unverified_context
    except AttributeError:
        pass
    else:
        ssl._create_default_https_context = _create_unverified_https_context

    resources = ['punkt', 'averaged_perceptron_tagger', 'punkt_tab','wordnet','averaged_perceptron_tagger_eng']
    for resource in resources:
        try:
            nltk.download(resource, quiet=True)
        except Exception as e:
            print(f"Error downloading {resource}: {str(e)}")


# This class  contains methods to humanize academic text, such as improving readability or
# simplifying complex language.
class AcademicTextHumanizer:
    """
    Transforms text into a more formal (academic) style:
      - Expands contractions
      - Adds academic transitions
      - Optionally converts some sentences to passive voice
      - Optionally replaces words with synonyms for more formality
    """

    def __init__(
        self,
        model_name='paraphrase-MiniLM-L6-v2',
        p_passive=0.2,
        p_synonym_replacement=0.3,
        p_academic_transition=0.3,
        seed=None
    ):
        if seed is not None:
            random.seed(seed)

        self.nlp = spacy.load("en_core_web_sm")
        self.model = SentenceTransformer(model_name)

        # Transformation probabilities
        self.p_passive = p_passive
        self.p_synonym_replacement = p_synonym_replacement
        self.p_academic_transition = p_academic_transition

        # Common academic transitions
        self.academic_transitions = [
            "Moreover,", "Additionally,", "Furthermore,", "Hence,",
            "Therefore,", "Consequently,", "Nevertheless,", "In addition,",
            "Subsequently,", "Accordingly,", "Conversely,", "However,",
            "Furthermore,", "Thus,", "To illustrate,", "For instance,",
            "Specifically,", "In contrast,", "Alternatively,", "Ultimately,",
            "Significantly,", "Notably,", "In conclusion,", "To summarize,"
        ]

    def humanize_text(self, text, use_passive=False, use_synonyms=False):
        doc = self.nlp(text)
        transformed_sentences = []

        for sent in doc.sents:
            sentence_str = sent.text.strip()

            # 1. Expand contractions
            sentence_str = self.expand_contractions(sentence_str)

            # 2. Possibly add academic transitions
            if random.random() < self.p_academic_transition:
                sentence_str = self.add_academic_transitions(sentence_str)

            # 3. Optionally convert to passive
            if use_passive and random.random() < self.p_passive:
                sentence_str = self.convert_to_passive(sentence_str)

            # 4. Optionally replace words with synonyms
            if use_synonyms and random.random() < self.p_synonym_replacement:
                sentence_str = self.replace_with_synonyms(sentence_str)

            transformed_sentences.append(sentence_str)

        return ' '.join(transformed_sentences)

    def expand_contractions(self, sentence):
        contraction_map = {
            "ain't": "am not", "aren't": "are not", "can't": "cannot", "can't've": "cannot have",
            "could've": "could have", "couldn't": "could not", "couldn't've": "could not have",
            "didn't": "did not", "doesn't": "does not", "don't": "do not", "hadn't": "had not",
            "hadn't've": "had not have", "hasn't": "has not", "haven't": "have not", "he'd": "he would",
            "he'd've": "he would have", "he'll": "he will", "he'll've": "he will have", "he's": "he is",
            "how'd": "how did", "how'd'y": "how do you", "how'll": "how will", "how's": "how is",
            "I'd": "I would", "I'd've": "I would have", "I'll": "I will", "I'll've": "I will have",
            "I'm": "I am", "I've": "I have", "isn't": "is not", "it'd": "it would",
            "it'd've": "it would have", "it'll": "it will", "it'll've": "it will have", "it's": "it is",
            "let's": "let us", "ma'am": "madam", "mayn't": "may not", "might've": "might have",
            "mightn't": "might not", "mightn't've": "might not have", "must've": "must have",
            "mustn't": "must not", "mustn't've": "must not have", "needn't": "need not",
            "needn't've": "need not have", "o'clock": "of the clock", "oughtn't": "ought not",
            "oughtn't've": "ought not have", "shan't": "shall not", "sha'n't": "shall not",
            "shan't've": "shall not have", "she'd": "she would", "she'd've": "she would have",
            "she'll": "she will", "she'll've": "she will have", "she's": "she is",
            "should've": "should have", "shouldn't": "should not", "shouldn't've": "should not have",
            "so's": "so is", "such's": "such is", "t's": "it is", "that'd": "that would",
            "that'd've": "that would have", "that's": "that is", "there'd": "there would",
            "there'd've": "there would have", "there's": "there is", "these's": "these is",
            "they'd": "they would", "they'd've": "they would have", "they'll": "they will",
            "they'll've": "they will have", "they're": "they are", "they've": "they have",
            "this's": "this is", "those's": "those is", "to've": "to have", "wasn't": "was not",
            "we'd": "we would", "we'd've": "we would have", "we'll": "we will", "we'll've": "we will have",
            "we're": "we are", "we've": "we have", "weren't": "were not", "what'll": "what will",
            "what'll've": "what will have", "what're": "what are", "what's": "what is",
            "what've": "what have", "when's": "when is", "when've": "when have", "where'd": "where did",
            "where's": "where is", "where've": "where have", "who'll": "who will", "who'll've": "who will have",
            "who's": "who is", "who've": "who have", "why's": "why is", "why've": "why have",
            "will've": "will have", "won't": "will not", "won't've": "will not have", "would've": "would have",
            "wouldn't": "would not", "wouldn't've": "would not have", "y'all": "you all",
            "y'all'd": "you all would", "y'all'd've": "you all would have", "y'all're": "you all are",
            "y'all've": "you all have", "you'd": "you would", "you'd've": "you would have",
            "you'll": "you will", "you'll've": "you will have", "you're": "you are", "you've": "you have"
        }
        
        # Sort contractions by length in descending order to prioritize longer matches
        sorted_contractions = sorted(contraction_map.keys(), key=len, reverse=True)

        # Use regular expression for more robust replacement, considering word boundaries
        def replace_contraction(match):
            contraction = match.group(0)
            return contraction_map.get(contraction.lower(), contraction)

        # Create a regex pattern that matches whole words for contractions
        pattern = r'(?:' + '|'.join(re.escape(c) for c in sorted_contractions) + r')'
        # Ensure 're' and 's' are handled carefully to not expand possessives or parts of words
        # This simple regex might be too aggressive for 's, we might need a more sophisticated approach for that.
        
        # The current implementation of word_tokenize followed by iterating and replacing will be maintained
        # but the contraction_map will be updated for more comprehensive coverage.
        
        tokens = word_tokenize(sentence)
        expanded_tokens = []
        for token in tokens:
            # Check for exact match with lowercased token for most contractions
            if token.lower() in contraction_map:
                expanded_tokens.append(contraction_map[token.lower()])
            else:
                # Handle cases like "n't" as suffixes more carefully
                replaced = False
                for contraction_suffix, expansion in [("n't", " not"), ("'s", " is"), ("'re", " are"), ("'ll", " will"), ("'ve", " have"), ("'d", " would"), ("'m", " am")]:
                    if token.lower().endswith(contraction_suffix) and len(token) > len(contraction_suffix):
                        root_word = token[:-len(contraction_suffix)]
                        # Preserve capitalization of the root word
                        if root_word[0].isupper():
                            expanded_tokens.append(root_word + expansion.capitalize())
                        else:
                            expanded_tokens.append(root_word + expansion)
                        replaced = True
                        break
                if not replaced:
                    expanded_tokens.append(token)
        return ' '.join(expanded_tokens)

    def add_academic_transitions(self, sentence):
        doc = self.nlp(sentence)
        # Avoid adding transition if the sentence already starts with one
        if any(sentence.lower().startswith(t.lower().replace(',', '').strip()) for t in self.academic_transitions):
            return sentence

        # More context-aware placement: try to find a suitable insertion point
        # For simplicity, let's try to insert after the first clause or at the beginning
        # A more advanced approach would involve deeper parsing.

        # Option 1: Insert at the beginning (current behavior, but with more choices)
        transition = random.choice(self.academic_transitions)
        return f"{transition} {sentence}"

        # Option 2 (more complex, consider for future): Find a comma or conjunction
        # for a mid-sentence insertion. This would require more sophisticated Spacy parsing
        # to ensure grammatical correctness and flow, which is beyond the scope of a
        # quick improvement. For now, we'll stick to beginning insertions with more choices.

    def convert_to_passive(self, sentence):
        doc = self.nlp(sentence)
        # A more robust passive conversion requires deeper linguistic analysis.
        # For this iteration, we'll focus on improving the existing pattern matching.
        
        # Find the main verb and its subject and direct object
        root = None
        for token in doc:
            if token.dep_ == "ROOT":
                root = token
                break
        
        if root and root.pos_ == "VERB":
            subject = None
            direct_object = None

            for child in root.children:
                if child.dep_ == "nsubj" or child.dep_ == "nsubjpass":
                    subject = child
                elif child.dep_ == "dobj":
                    direct_object = child

            if subject and direct_object:  # Active voice candidate
                # Get the base form of the verb
                verb_lemma = root.lemma_
                # Use TextBlob to conjugate the verb to its past participle form
                blob = TextBlob(root.text)
                try:
                    # Attempt to get the past participle form. TextBlob's `lemmas` can sometimes provide this
                    # but a direct `to_passive()` or similar is not available for full sentence conversion.
                    # For now, we'll try a simpler heuristic with TextBlob's tags or rely on manual rules if that fails.
                    
                    # A more robust approach would be to use a dedicated verb conjugation library.
                    # For this iteration, let's try to infer from the original verb tense.

                    # Simple heuristic: assume 'be' verb form based on original verb tense (simplified)
                    # This is still a simplification but better than 'is ...ed'
                    be_verb = "is" # Default for present simple
                    if "VBD" in root.tag_: # Past tense
                        be_verb = "was"
                    elif "VBG" in root.tag_: # Present participle
                        be_verb = "is being"
                    elif "VBN" in root.tag_: # Past participle (already passive or perfect tense)
                        be_verb = "has been" # This might not be fully accurate for perfect tenses
                    elif "VBZ" in root.tag_: # 3rd person singular present
                        be_verb = "is"
                    elif "VBP" in root.tag_: # Non-3rd person singular present
                        be_verb = "are"
                    
                    # Get the past participle of the main verb
                    # TextBlob's `lemmas` doesn't directly provide past participle.
                    # We'll use a simple rule for regular verbs and rely on Spacy's lemma for irregulars.
                    past_participle = root.lemma_ + "ed" # Simplified for regular verbs
                    if root.tag_.startswith("VB"): # If it's a verb tag
                        # For irregular verbs, Spacy's lemma is often the base form.
                        # TextBlob's word.lemmatize('v') can also give the base form.
                        # We need to find the past participle form. This often requires a dictionary or more rules.
                        # Given the constraints, a simple heuristic for irregular verbs is hard without a full conjugator.
                        # Let's stick to a very basic rule for now and note it as a future improvement.
                        pass

                    # For now, we'll keep the simplified `passive_verb` construction
                    passive_verb = f"{be_verb} {root.text}ed" # Still very simplified
                    
                    # If the root verb is irregular, TextBlob doesn't directly help much for past participle
                    # A robust solution would involve a comprehensive verb conjugation library.

                    # Reconstruct sentence with passive structure
                    active_chunk_start = min(subject.i, root.i, direct_object.i)
                    active_chunk_end = max(subject.i, root.i, direct_object.i)
                    
                    # Construct the passive chunk more carefully, considering original sentence parts
                    passive_chunk_tokens = []
                    
                    # Add tokens before the active chunk
                    for token in doc[:active_chunk_start]:
                        passive_chunk_tokens.append(token.text_with_ws)
                    
                    # Add the direct object
                    passive_chunk_tokens.append(direct_object.text_with_ws)

                    # Add the 'be' verb and past participle
                    # This part needs to be more intelligent about tense and agreement
                    passive_chunk_tokens.append(f"{be_verb} ") # Simple 'is' or 'was' for now
                    
                    # Attempt to get past participle from TextBlob or simple rule
                    conjugated_verb = TextBlob(root.text).words[0].conjugate(tense='past_participle')
                    passive_chunk_tokens.append(f"{conjugated_verb} ")

                    passive_chunk_tokens.append(f"by {subject.text_with_ws}")

                    # Add tokens after the active chunk
                    for token in doc[active_chunk_end + 1:]:
                        passive_chunk_tokens.append(token.text_with_ws)

                    sentence = "".join(passive_chunk_tokens).strip()
                    
                except Exception as e:
                    # Fallback to original sentence if TextBlob conjugation fails
                    print(f"TextBlob conjugation error: {e}")
                    pass

        return sentence

    def replace_with_synonyms(self, sentence):
        tokens = word_tokenize(sentence)
        pos_tags = nltk.pos_tag(tokens)

        new_tokens = []
        for (word, pos) in pos_tags:
            # Introduce a more dynamic replacement probability based on word length or frequency
            # For simplicity, let's make longer words slightly more likely to be replaced
            # and also ensure the word is not a common stop word.
            
            # NLTK stopwords need to be downloaded if not present
            try:
                nltk.data.find('corpora/stopwords')
            except nltk.downloader.DownloadError:
                nltk.download('stopwords', quiet=True)
            from nltk.corpus import stopwords
            stop_words = set(stopwords.words('english'))
            
            replacement_chance = self.p_synonym_replacement # Base probability
            if len(word) > 6: # Longer words get a slightly higher chance
                replacement_chance *= 1.2
            if word.lower() in stop_words: # Don't replace stop words
                replacement_chance = 0

            if pos.startswith(('J', 'N', 'V', 'R')) and wordnet.synsets(word) and random.random() < replacement_chance:
                synonyms = self._get_synonyms(word, pos)
                if synonyms:
                    best_synonym = self._select_closest_synonym(word, synonyms)
                    # Only replace if a good synonym is found and it's sufficiently different
                    if best_synonym and best_synonym.lower() != word.lower():
                        new_tokens.append(best_synonym)
                    else:
                        new_tokens.append(word)
                else:
                    new_tokens.append(word)
            else:
                new_tokens.append(word)

        return ' '.join(new_tokens)

    def _get_synonyms(self, word, pos):
        wn_pos = None
        if pos.startswith('J'):
            wn_pos = wordnet.ADJ
        elif pos.startswith('N'):
            wn_pos = wordnet.NOUN
        elif pos.startswith('R'):
            wn_pos = wordnet.ADV
        elif pos.startswith('V'):
            wn_pos = wordnet.VERB

        synonyms = set()
        for syn in wordnet.synsets(word, pos=wn_pos):
            for lemma in syn.lemmas():
                lemma_name = lemma.name().replace('_', ' ')
                if lemma_name.lower() != word.lower():
                    synonyms.add(lemma_name)
        return list(synonyms)

    def _select_closest_synonym(self, original_word, synonyms):
        if not synonyms:
            return None
        original_emb = self.model.encode(original_word, convert_to_tensor=True)
        synonym_embs = self.model.encode(synonyms, convert_to_tensor=True)
        cos_scores = util.cos_sim(original_emb, synonym_embs)[0]
        max_score_index = cos_scores.argmax().item()
        max_score = cos_scores[max_score_index].item()
        if max_score >= 0.5:
            return synonyms[max_score_index]
        return None