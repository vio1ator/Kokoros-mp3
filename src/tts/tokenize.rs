use crate::tts::vocab::VOCAB;
pub fn tokenize(text: &str) -> Vec<i64> {
    text.chars()
        .filter_map(|c| VOCAB.get(&c))
        .map(|&idx| idx as i64)
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tokenize() {
        let text = "Hello!";
        let tokens = tokenize(text);

        // Expected tokens based on the vocabulary mapping defined in get_vocab()
        let expected = vec![24, 47, 54, 54, 57, 5];

        assert_eq!(tokens, expected);

        // Test empty string
        let empty = "";
        let empty_tokens = tokenize(empty);
        assert!(empty_tokens.is_empty());

        // Test punctuation
        let punct = "...";
        let punct_tokens = tokenize(punct);
        assert_eq!(punct_tokens.len(), 3);
    }
}
