using LLama;

namespace EmbeddingsApi2;
using LLama;

class LlamaLocalEmbedder : IEmbedder
{
    private readonly LLamaEmbedder _embedder;

    public LlamaLocalEmbedder(LLamaEmbedder embedder)
    {
        _embedder = embedder;
    }

    public async Task<float[]> GetEmbeddingsAsync(string text)
    {
        var embeddings = await _embedder.GetEmbeddings(text);
        return embeddings;
    }
}
