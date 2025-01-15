namespace EmbeddingsApi2
{
    
    /// <summary>
    /// Provides a cache for the model embedder to manage its state and availability.
    /// </summary>
    static class ModelCache
    {
        
        /// <summary>
        /// Gets or sets the current embedder instance.
        /// </summary>
        /// <value>
        /// An implementation of <see cref="IEmbedder"/>, or <c>null</c> if no embedder is loaded.
        /// </value>
        public static IEmbedder? Embedder { get; set; }
        
        /// <summary>
        /// Gets a value indicating whether a model is currently loaded in the cache.
        /// </summary>
        /// <value>
        /// <c>true</c> if an embedder is loaded; otherwise, <c>false</c>.
        /// </value>
        public static bool IsModelLoaded => Embedder != null;
    }
}

