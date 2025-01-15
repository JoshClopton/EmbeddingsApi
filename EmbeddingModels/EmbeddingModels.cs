namespace EmbeddingsApi2.EmbeddingModels
{
    /// <summary>
    /// Represents a request for generating embeddings.
    /// </summary>
    ///  /// <param name="Texts">A list of input texts for which embeddings are to be generated.</param>
    public record EmbeddingRequest(List<string> Texts);
    /// <summary>
    /// Represents a response containing embeddings.
    /// </summary>
    /// <param name="Embeddings">A list of embeddings, where each embedding is an array of floating-point values corresponding to an input text.</param>
    public record EmbeddingResponse(List<float[]> Embeddings);
}



