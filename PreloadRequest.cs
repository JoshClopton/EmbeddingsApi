namespace EmbeddingsApi2;

public record PreloadRequest(
    string? ApiKey,
    string? ModelId,         // e.g. "TheBloke/dolphin-2.2.1-mistral-7B-GGUF"
    string? ModelFormat,     // e.g. "gguf" for LLaMA, or "onnx" for MiniLM 
    string? SourceFilename,  // e.g. "mymodel.gguf" or "all-minilm-l6-v2.onnx"
    string? OutputFilename,   // local file path to store downloaded model
    string? VocabularyPath
);