namespace EmbeddingsApi2;

static class ModelCache
{
    public static IEmbedder? Embedder { get; set; }
    public static bool IsModelLoaded => Embedder != null;
}