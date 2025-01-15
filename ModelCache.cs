namespace EmbeddingsApi2
{
    /// <summary>
    /// Provides a cache for the model embedder to manage its state and availability.
    /// </summary>
    static class ModelCache
    {
        public static IEmbedder? Embedder { get; set; }
        public static bool IsModelLoaded => Embedder != null;
    }
}

