"""
Unit tests for text_preprocess module in src/
"""

import sys
import os
import pytest

src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
if src_path not in sys.path:
    sys.path.insert(0, src_path)

from text_preprocess import (
    replace_words,
    remove_punctuation,
    remove_stopwords,
    lem,
    lemmatize,
    appos,
    stoplist,
)


class TestTextPreprocess:
    def test_appos_dictionary_loaded(self):
        """Verify the contraction dictionary is populated."""
        assert isinstance(appos, dict)
        assert "it's" in appos
        assert appos["it's"] == "it is"
        assert "can't" in appos
        assert appos["can't"] == "cannot"

    def test_replace_words_contractions(self):
        """Test expansion of common English contractions."""
        input_text = "it's obvious that you don't know what you're doing"
        expected = "it is obvious that you do not know what you are doing"
        assert replace_words(input_text) == expected

    def test_replace_words_case_insensitive(self):
        """Test that contractions with uppercase letters are expanded."""
        input_text = "Ain't no mountain high enough"
        assert "am not" in replace_words(input_text).lower()

    def test_replace_words_no_contractions(self):
        """Test that text without contractions remains unchanged."""
        input_text = "The quick brown fox jumps over the lazy dog"
        assert replace_words(input_text) == input_text

    def test_remove_punctuation(self):
        """Test removal of all ASCII punctuation characters."""
        input_text = "Hello, world! (This: is a test; with punctuation?)"
        expected = "Hello world This is a test with punctuation"
        assert remove_punctuation(input_text) == expected

    def test_remove_stopwords_preserves_not(self):
        """Test that stopwords are removed, but 'not' is preserved for sarcasm context."""
        assert "not" not in stoplist, "'not' must be excluded from stopwords for sarcasm analysis"
        input_text = "this is not a sarcastic headline at all"
        result = remove_stopwords(input_text)
        assert "not" in result.split()
        assert "this" not in result.split()
        assert "is" not in result.split()

    def test_lem_pos_tagging(self):
        """Test POS-aware lemmatization on nouns and verbs."""
        input_text = "cats are running"
        result = lem(input_text)
        tokens = result.split()
        assert "cat" in tokens
        assert "run" in tokens or "running" in tokens

    def test_lemmatize_word_level(self):
        """Test word-level lemmatization (verifying fix from character-level bug)."""
        result = lemmatize("cats running")
        assert "cat" in result.split()
