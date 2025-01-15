namespace EmbeddingsApi2
{
    
    /// <summary>
    /// Represents a request to preload a machine learning model for embedding generation.
    /// </summary>
    /// <param name="ApiKey">The API key used to authenticate requests to the model source, if required.</param>
    /// <param name="ModelId">The identifier of the model to load, e.g., "TheBloke/dolphin-2.2.1-mistral-7B-GGUF".</param>
    /// <param name="ModelFormat">The format of the model, e.g., "gguf" for LLaMA models or "onnx" for MiniLM models.</param>
    /// <param name="SourceFilename">The name of the source file for the model, e.g., "mymodel.gguf" or "all-minilm-l6-v2.onnx".</param>
    /// <param name="OutputFilename">The local file path where the downloaded model should be stored.</param>
    /// <param name="VocabularyPath">The file path where the vocabulary file for the model is located. 
    /// This file typically contains a mapping of words or tokens to their corresponding IDs, which is essential for processing and tokenizing text inputs in NLP models.</param>
    public record PreloadRequest(
        string? ApiKey,
        string? ModelId,
        string? ModelFormat,
        string? SourceFilename,
        string? OutputFilename,
        string? VocabularyPath
    );
}