namespace EmbeddingsApi2;

public interface IEmbedder
{
    Task<float[]> GetEmbeddingsAsync(string text);
}