import pytest
import tiktoken
import os

from minbpe import BasicTokenizer, RegexTokenizer, GPT4Tokenizer


# -------------------------------------------------------
# Common test data
# -------------------------------------------------------

# A few strings to test the tokenizers on
test_strings = [
    "",  # empty string
    "?",  # single character
    "hello world!!!? (안녕하세요!) lol123 😉",  # fun small string
    "FILE:taylorswift.txt",  # FILE: is handled as a special string in unpack()
]


def unpack(text):
    """
    We do this because `pytest -v .` prints the arguments to console,
    and we don't want to print the entire contents of the file.
    """
    if text.startswith("FILE:"):
        dirname = os.path.dirname(os.path.abspath(__file__))
        filepath = os.path.join(dirname, text[5:])
        with open(filepath, "r", encoding="utf-8") as f:
            contents = f.read()
        return contents
    else:
        return text


specials_string = """
<|endoftext|>Hello world this is one document
<|endoftext|>And this is another document
<|endoftext|><|fim_prefix|>And this one has<|fim_suffix|> tokens.<|fim_middle|> FIM
<|endoftext|>Last document!!! 👋<|endofprompt|>
""".strip()


special_tokens = {
    "<|endoftext|>": 100257,
    "<|fim_prefix|>": 100258,
    "<|fim_middle|>": 100259,
    "<|fim_suffix|>": 100260,
    "<|endofprompt|>": 100276,
}


llam_text = """
<|endoftext|>The llama (/ˈlɑːmə/; Spanish pronunciation: [ˈʎama] or [ˈʝama]) (Lama glama) is a domesticated South American camelid...
<|fim_prefix|>In Aymara mythology, llamas are important beings.<|fim_suffix|> where they come from at the end of time.<|fim_middle|> llamas will return to the water springs and ponds<|endofprompt|>
""".strip()


# -------------------------------------------------------
# TESTS
# -------------------------------------------------------

# Test encode/decode identity
@pytest.mark.parametrize("tokenizer_factory", [BasicTokenizer, RegexTokenizer, GPT4Tokenizer])
@pytest.mark.parametrize("text", test_strings)
def test_encode_decode_identity(tokenizer_factory, text):
    text = unpack(text)
    tokenizer = tokenizer_factory()
    ids = tokenizer.encode(text)
    decoded = tokenizer.decode(ids)
    assert text == decoded


# Test that our GPT4 tokenizer matches official tiktoken
@pytest.mark.parametrize("text", test_strings)
def test_gpt4_tiktoken_equality(text):
    text = unpack(text)
    tokenizer = GPT4Tokenizer()
    enc = tiktoken.get_encoding("cl100k_base")

    tiktoken_ids = enc.encode(text)
    gpt4_tokenizer_ids = tokenizer.encode(text)

    assert gpt4_tokenizer_ids == tiktoken_ids


# Test special token handling
def test_gpt4_tiktoken_equality_special_tokens():
    tokenizer = GPT4Tokenizer()
    enc = tiktoken.get_encoding("cl100k_base")

    tiktoken_ids = enc.encode(specials_string, allowed_special="all")
    gpt4_tokenizer_ids = tokenizer.encode(specials_string, allowed_special="all")

    assert gpt4_tokenizer_ids == tiktoken_ids


# Wikipedia BPE reference example
@pytest.mark.parametrize("tokenizer_factory", [BasicTokenizer, RegexTokenizer])
def test_wikipedia_example(tokenizer_factory):
    """
    According to Wikipedia:
    Running BPE on "aaabdaaabac" for 3 merges results in:
    "XdXac"

    where:
    Z = aa (256)
    Y = ab (257)
    X = ZY (258)

    Expected ids: [258, 100, 258, 97, 99]
    """
    tokenizer = tokenizer_factory()
    text = "aaabdaaabac"

    tokenizer.train(text, 256 + 3)
    ids = tokenizer.encode(text)

    assert ids == [258, 100, 258, 97, 99]
    assert tokenizer.decode(ids) == text


# Test save/load functionality
@pytest.mark.parametrize("special_tokens", [{}, special_tokens])
def test_save_load(special_tokens):
    text = llam_text

    tokenizer = RegexTokenizer()
    tokenizer.train(text, 256 + 64)
    tokenizer.register_special_tokens(special_tokens)

    # Verify encode/decode identity
    assert tokenizer.decode(tokenizer.encode(text, allowed_special="all")) == text

    ids = tokenizer.encode_ordinary(text)

    # Save tokenizer
    tokenizer.save("test_tokenizer_tmp")

    # Reload tokenizer
    tokenizer = RegexTokenizer()
    tokenizer.load("test_tokenizer_tmp.model")

    # Verify functionality after load
    assert tokenizer.decode(ids) == text
    assert tokenizer.decode(tokenizer.encode(text, allowed_special="all")) == text
    assert tokenizer.encode_ordinary(text) == ids

    # Cleanup temp files
    for file in ["test_tokenizer_tmp.model", "test_tokenizer_tmp.vocab"]:
        if os.path.exists(file):
            os.remove(file)


if __name__ == "__main__":
    pytest.main()
