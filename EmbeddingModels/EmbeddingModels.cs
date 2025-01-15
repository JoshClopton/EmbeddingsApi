namespace EmbeddingsApi2.EmbeddingModels
{
    public record EmbeddingRequest(List<string> Texts);
    public record EmbeddingResponse(List<float[]> Embeddings);
}



