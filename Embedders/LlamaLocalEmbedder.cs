namespace EmbeddingsApi2.Embedders
{
    using LLama;
    using System.Threading.Tasks;

    /// <summary>
    /// Provides functionality to generate embeddings locally using the LLamaEmbedder.
    /// </summary>
    public class LlamaLocalEmbedder : IEmbedder
    {
        #region Public-Members

        #endregion
        
        #region Private-Members

        private readonly LLamaEmbedder _Embedder;

        #endregion

        #region Constructors-and-Factories

        /// <summary>
        /// Initializes a new instance of the <see cref="LlamaLocalEmbedder"/> class.
        /// </summary>
        /// <param name="embedder">The LLamaEmbedder instance to use for generating embeddings.</param>
        public LlamaLocalEmbedder(LLamaEmbedder embedder)
        {
            _Embedder = embedder;
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
            var embeddings = await _Embedder.GetEmbeddings(text);
            return embeddings;
        }

        #endregion

        #region Private-Methods
        
        #endregion
    }
}