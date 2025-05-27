import re

def clean_text(text: str) -> str:
    """
    Cleans a given text string by performing several normalization steps.

    The cleaning process includes:
    1.  Removing HTML tags.
    2.  Removing URLs.
    3.  Removing special characters (keeps alphanumeric and spaces).
    4.  Replacing multiple consecutive spaces with a single space.
    5.  Trimming leading and trailing whitespace from the entire text.
    6.  Normalizing whitespace (ensuring single spaces between words).

    Args:
        text (str): The input string to be cleaned.

    Returns:
        str: The cleaned text string.
    """
    # Remove HTML tags
    text = re.sub(r'<[^>]*?>', '', text)
    # Remove URLs
    text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
    # Remove special characters (retaining only alphanumeric characters and spaces)
    text = re.sub(r'[^a-zA-Z0-9 ]', '', text)
    # Replace multiple spaces with a single space
    text = re.sub(r'\s{2,}', ' ', text)
    # Trim leading and trailing whitespace
    text = text.strip()
    # Remove extra whitespace that might have been introduced or left over, ensuring single spaces between words
    text = ' '.join(text.split())
    return text