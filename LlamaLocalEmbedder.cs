using LLama;

namespace EmbeddingsApi2;
using LLama;

public class LlamaLocalEmbedder : IEmbedder
{
    private readonly LLamaEmbedder _embedder;

    public LlamaLocalEmbedder(LLamaEmbedder embedder)
    {
        _embedder = embedder;
    }

    public async Task<float[]> GetEmbeddingsAsync(string text)
    {
        // LLamaEmbedder doesn't have an async method,
        // but we can wrap it in Task.Run if needed
        return await Task.Run(() => _embedder.GetEmbeddings(text));
    }
}
