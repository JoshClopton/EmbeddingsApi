namespace EmbeddingsApi2
{
    using System.Threading.Tasks;
    /// <summary>
    /// Defines a contract for embedding generation functionality.
    /// </summary>
    public interface IEmbedder
    {
        /// <summary>
        /// Asynchronously generates embeddings for the specified text.
        /// </summary>
        /// <param name="text">The input text to generate embeddings for.</param>
        /// <returns>A task that represents the asynchronous operation. 
        /// The task result contains an array of floats representing the embeddings.</returns>
        Task<float[]> GetEmbeddingsAsync(string text);
    }
}