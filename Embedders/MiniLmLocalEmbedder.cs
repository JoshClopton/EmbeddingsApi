namespace EmbeddingsApi2.Embedders
{
    using Microsoft.ML.OnnxRuntime;
    using Microsoft.ML.OnnxRuntime.Tensors;

    /// <summary>
    /// Provides functionality to generate embeddings using a locally stored MiniLM ONNX model.
    /// </summary>
    class MiniLmLocalEmbedder : IEmbedder
    {
        #region Public-Members
        
        #endregion
        
        #region Private-Members
        private readonly InferenceSession _session;
        private readonly Dictionary<string, int> _vocabulary;
        private const int MaxLength = 256;
        #endregion
        
        #region Constructors-and-Factories
        /// <summary>
        /// Initializes a new instance of the <see cref="MiniLmLocalEmbedder"/> class.
        /// </summary>
        /// <param name="modelPath">The file path to the ONNX model.</param>
        /// <param name="vocabularyPath">The file path to the vocabulary file.</param>
        public MiniLmLocalEmbedder(string modelPath, string vocabularyPath)
        {
            _session = new InferenceSession(modelPath);
            _vocabulary = LoadVocabulary(vocabularyPath);
        }
        #endregion
        
        #region Public-Methods
        /// <summary>
        /// Asynchronously generates embeddings for the provided text.
        /// </summary>
        /// <param name="text">The input text to generate embeddings for.</param>
        /// <returns>A task that represents the asynchronous operation. 
        /// The task result contains an array of floats representing the embeddings.</returns>
        public async Task<float[]> GetEmbeddingsAsync(string text)
        {
            // Tokenize the input text
            var (tokenIds, attentionMask) = Tokenize(text);

            // Create token_type_ids (all zeros, same length as input_ids)
            var tokenTypeIds = new long[tokenIds.Count];
            Array.Fill(tokenTypeIds, 0L);

            // Create input tensors
            var inputTensor = new DenseTensor<long>(tokenIds.ToArray(), new[] { 1, tokenIds.Count });
            var maskTensor = new DenseTensor<long>(attentionMask.ToArray(), new[] { 1, attentionMask.Count });
            var tokenTypeTensor = new DenseTensor<long>(tokenTypeIds, new[] { 1, tokenTypeIds.Length });

            var inputs = new List<NamedOnnxValue>
            {
                NamedOnnxValue.CreateFromTensor("input_ids", inputTensor),
                NamedOnnxValue.CreateFromTensor("attention_mask", maskTensor),
                NamedOnnxValue.CreateFromTensor("token_type_ids", tokenTypeTensor)
            };

            // Run inference
            using var results = _session.Run(inputs);

            // Extract embeddings from the model output
            // Note: The exact output tensor name might vary, you might need to check the model's output names
            var embeddings = results.First().AsTensor<float>().ToArray();

            return embeddings;
        }
        #endregion
        
        #region Private-Methods
        /// <summary>
        /// Loads the vocabulary from the specified file path.
        /// </summary>
        /// <param name="path">The file path to the vocabulary file.</param>
        /// <returns>A dictionary containing words as keys and their token IDs as values.</returns>
        private Dictionary<string, int> LoadVocabulary(string path)
        {
            var vocabulary = new Dictionary<string, int>();
            var lines = File.ReadAllLines(path);
            for (int i = 0; i < lines.Length; i++)
            {
                vocabulary[lines[i].Trim()] = i;
            }

            return vocabulary;
        }
        /// <summary>
        /// Tokenizes the input text into token IDs and attention masks.
        /// </summary>
        /// <param name="text">The input text to tokenize.</param>
        /// <returns>A tuple containing a list of token IDs and a list of attention masks.</returns>
        private (List<long> TokenIds, List<long> AttentionMask) Tokenize(string text)
        {
            // Initialize with special tokens
            var tokenIds = new List<long> { _vocabulary["[CLS]"] };
            var attentionMask = new List<long> { 1 };

            // Basic tokenization
            var words = text.Split(' ', StringSplitOptions.RemoveEmptyEntries);
            foreach (var word in words)
            {
                // If the word exists in vocabulary, add it
                if (_vocabulary.TryGetValue(word.ToLower(), out int tokenId))
                {
                    tokenIds.Add(tokenId);
                    attentionMask.Add(1);
                }
                else
                {
                    // WordPiece tokenization for unknown words
                    var subwords = TokenizeWord(word);
                    tokenIds.AddRange(subwords);
                    attentionMask.AddRange(Enumerable.Repeat(1L, subwords.Count));
                }

                // Check if we're approaching max length (leaving room for [SEP])
                if (tokenIds.Count >= MaxLength - 1)
                    break;
            }

            // Add [SEP] token
            tokenIds.Add(_vocabulary["[SEP]"]);
            attentionMask.Add(1);

            // Pad if necessary (but keep actual length if shorter than MaxLength)
            int actualLength = tokenIds.Count;
            while (tokenIds.Count < MaxLength)
            {
                tokenIds.Add(_vocabulary["[PAD]"]);
                attentionMask.Add(0);
            }

            // Trim to actual used length
            return (
                tokenIds.Take(actualLength).ToList(),
                attentionMask.Take(actualLength).ToList()
            );
        }
        /// <summary>
        /// Tokenizes a single word into subword token IDs.
        /// </summary>
        /// <param name="word">The word to tokenize.</param>
        /// <returns>A list of token IDs representing the subword tokens.</returns>
        private List<long> TokenizeWord(string word)
        {
            var tokens = new List<long>();
            var currentToken = "";

            foreach (char c in word.ToLower())
            {
                currentToken += c;
                string prefix = currentToken;

                // Try with "##" prefix for subwords
                if (tokens.Count > 0)
                {
                    prefix = "##" + currentToken;
                }

                if (_vocabulary.TryGetValue(prefix, out int tokenId))
                {
                    tokens.Add(tokenId);
                    currentToken = "";
                }
            }

            // If we couldn't tokenize the word, use UNK token
            if (tokens.Count == 0 || currentToken.Length > 0)
            {
                tokens.Add(_vocabulary["[UNK]"]);
            }

            return tokens;
        }
        #endregion
    }
}