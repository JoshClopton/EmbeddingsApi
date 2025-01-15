using Microsoft.AspNetCore.Builder;
using Microsoft.AspNetCore.Hosting;
using Microsoft.Extensions.Hosting;
using Microsoft.AspNetCore.Http;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using System.Text.Json;
using System.Text.Json.Serialization;
using System.Threading.Tasks;
using System;
using System.Collections.Generic;
using EmbeddingsApi2;
using RestWrapper;

// LLamaSharp for GGUF models
using LLama;
using LLama.Common;

// Minimal API Program
var builder = WebApplication.CreateBuilder(args);

var app = builder.Build();

// 1) GET /health – Check if the service is running
app.MapGet("/health", () => Results.Ok("Service is running!"));

app.MapPost("/preload", async (PreloadRequest req) =>
{
    try
    {
        Console.WriteLine($"[preload] Received request to load model: {req.ModelId}");

        // Initialize HuggingFaceClient (needed for both GGUF and ONNX)
        var logging = new SyslogLogging.LoggingModule();
        var serializer = new Serializer();
        var huggingFaceClient = new HuggingFaceClient(logging, serializer, req.ApiKey ?? "");

        // Step 1: Download the appropriate model if needed
        string localModelPath = Path.Combine(Directory.GetCurrentDirectory(), req.OutputFilename ?? "model.bin");
        
        if (req.ModelFormat?.Equals("gguf", StringComparison.OrdinalIgnoreCase) == true)
        {
            if (!File.Exists(localModelPath))
            {
                Console.WriteLine("[preload] Downloading GGUF model file...");
                bool success = await huggingFaceClient.DownloadGguf(
                    req.ModelId!,
                    req.SourceFilename ?? "model.gguf",  // Make sure to pass SourceFilename in request
                    localModelPath,
                    CancellationToken.None
                );

                if (!success)
                {
                    return Results.BadRequest("Failed to download GGUF model file.");
                }
            }
        }
        else if (req.ModelFormat?.Equals("onnx", StringComparison.OrdinalIgnoreCase) == true)
        {
            // Existing ONNX download logic...
            if (!File.Exists(localModelPath))
            {
                Console.WriteLine("[preload] Downloading ONNX model file...");
                bool success = await huggingFaceClient.DownloadOnnx(
                    req.ModelId!,
                    req.SourceFilename ?? "model.onnx",
                    localModelPath,
                    CancellationToken.None
                );

                if (!success)
                {
                    return Results.BadRequest("Failed to download ONNX model file.");
                }
            }
        }
        else
        {
            return Results.BadRequest("Invalid model format. Supported formats: gguf, onnx");
        }

        // Step 2: Load the model
        IEmbedder embedder;
        if (req.ModelFormat?.Equals("gguf", StringComparison.OrdinalIgnoreCase) == true)
        {
            Console.WriteLine("[preload] Loading GGUF model with LLamaSharp...");
            var mParams = new ModelParams(localModelPath)
            {
                EmbeddingMode = true
            };
            var weights = LLamaWeights.LoadFromFile(mParams);
            embedder = new LlamaLocalEmbedder(new LLamaEmbedder(weights, mParams));
        }
        else // ONNX
        {
            Console.WriteLine("[preload] Loading ONNX model...");
            string localVocabPath = Path.Combine(Directory.GetCurrentDirectory(), "vocab.txt");
            
            if (!File.Exists(localVocabPath))
            {
                Console.WriteLine("[preload] Downloading vocabulary file...");
                bool success = await huggingFaceClient.DownloadVocabulary(
                    req.ModelId!,
                    "vocab.txt",
                    localVocabPath,
                    CancellationToken.None
                );

                if (!success)
                {
                    return Results.BadRequest("Failed to download vocabulary file.");
                }
            }

            embedder = new MiniLmLocalEmbedder(localModelPath, localVocabPath);
        }

        // Store embedder in ModelCache
        ModelCache.Embedder = embedder;
        Console.WriteLine("[preload] Model loaded successfully.");

        return Results.Ok("Model preloaded successfully!");
    }
    catch (Exception ex)
    {
        Console.WriteLine($"[preload] Error loading model: {ex.Message}");
        return Results.BadRequest($"Error: {ex.Message}");
    }
});

// 3) POST /embeddings – Generate embeddings
app.MapPost("/embeddings", async (EmbeddingRequest req) =>
{
    if (!ModelCache.IsModelLoaded)
    {
        return Results.BadRequest("Model is not loaded. Call /preload first.");
    }

    try
    {
        var embedder = ModelCache.Embedder!;
        var embeddingsList = new List<float[]>();

        foreach (var text in req.Texts)
        {
            float[] emb = await embedder.GetEmbeddingsAsync(text);
            embeddingsList.Add(emb);
        }

        return Results.Ok(new EmbeddingResponse(embeddingsList));
    }
    catch (Exception ex)
    {
        Console.WriteLine($"[embeddings] Error generating embeddings: {ex.Message}");
        return Results.BadRequest($"Error generating embeddings: {ex.Message}");
    }
});

// Run the app
app.Run();

// Define helper classes and interfaces
record EmbeddingRequest(List<string> Texts);
record EmbeddingResponse(List<float[]> Embeddings);
interface IEmbedder
{
    Task<float[]> GetEmbeddingsAsync(string text);
}

static class ModelCache
{
    public static IEmbedder? Embedder { get; set; }
    public static bool IsModelLoaded => Embedder != null;
}

// LLaMA Embedder (for GGUF models)
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

class MiniLmLocalEmbedder : IEmbedder
{
    private readonly InferenceSession _session;
    private readonly Dictionary<string, int> _vocabulary;
    private const int MaxLength = 256;

    public MiniLmLocalEmbedder(string modelPath, string vocabularyPath)
    {
        _session = new InferenceSession(modelPath);
        _vocabulary = LoadVocabulary(vocabularyPath);
    }

    public async Task<float[]> GetEmbeddingsAsync(string text)
    {
        // Tokenize the input text
        var (tokenIds, attentionMask) = Tokenize(text);

        // Create token_type_ids (all zeros, same length as input_ids)
        var tokenTypeIds = new long[tokenIds.Count];
        Array.Fill(tokenTypeIds, 0L);

        // Create input tensors
        var inputTensor = new DenseTensor<long>(tokenIds.ToArray(), new[] { 1, tokenIds.Count });
        var maskTensor = new DenseTensor<long>(attentionMask.ToArray(), new[] { 1, attentionMask.Count });
        var tokenTypeTensor = new DenseTensor<long>(tokenTypeIds, new[] { 1, tokenTypeIds.Length });

        var inputs = new List<NamedOnnxValue>
        {
            NamedOnnxValue.CreateFromTensor("input_ids", inputTensor),
            NamedOnnxValue.CreateFromTensor("attention_mask", maskTensor),
            NamedOnnxValue.CreateFromTensor("token_type_ids", tokenTypeTensor)
        };

        // Run inference
        using var results = _session.Run(inputs);

        // Extract embeddings from the model output
        // Note: The exact output tensor name might vary, you might need to check the model's output names
        var embeddings = results.First().AsTensor<float>().ToArray();
        
        return embeddings;
    }

    private Dictionary<string, int> LoadVocabulary(string path)
    {
        var vocabulary = new Dictionary<string, int>();
        var lines = File.ReadAllLines(path);
        for (int i = 0; i < lines.Length; i++)
        {
            vocabulary[lines[i].Trim()] = i;
        }
        return vocabulary;
    }

    private (List<long> TokenIds, List<long> AttentionMask) Tokenize(string text)
    {
        // Initialize with special tokens
        var tokenIds = new List<long> { _vocabulary["[CLS]"] };
        var attentionMask = new List<long> { 1 };

        // Basic tokenization
        var words = text.Split(' ', StringSplitOptions.RemoveEmptyEntries);
        foreach (var word in words)
        {
            // If the word exists in vocabulary, add it
            if (_vocabulary.TryGetValue(word.ToLower(), out int tokenId))
            {
                tokenIds.Add(tokenId);
                attentionMask.Add(1);
            }
            else
            {
                // WordPiece tokenization for unknown words
                var subwords = TokenizeWord(word);
                tokenIds.AddRange(subwords);
                attentionMask.AddRange(Enumerable.Repeat(1L, subwords.Count));
            }

            // Check if we're approaching max length (leaving room for [SEP])
            if (tokenIds.Count >= MaxLength - 1)
                break;
        }

        // Add [SEP] token
        tokenIds.Add(_vocabulary["[SEP]"]);
        attentionMask.Add(1);

        // Pad if necessary (but keep actual length if shorter than MaxLength)
        int actualLength = tokenIds.Count;
        while (tokenIds.Count < MaxLength)
        {
            tokenIds.Add(_vocabulary["[PAD]"]);
            attentionMask.Add(0);
        }

        // Trim to actual used length
        return (
            tokenIds.Take(actualLength).ToList(),
            attentionMask.Take(actualLength).ToList()
        );
    }

    private List<long> TokenizeWord(string word)
    {
        var tokens = new List<long>();
        var currentToken = "";

        foreach (char c in word.ToLower())
        {
            currentToken += c;
            string prefix = currentToken;
            
            // Try with "##" prefix for subwords
            if (tokens.Count > 0)
            {
                prefix = "##" + currentToken;
            }

            if (_vocabulary.TryGetValue(prefix, out int tokenId))
            {
                tokens.Add(tokenId);
                currentToken = "";
            }
        }

        // If we couldn't tokenize the word, use UNK token
        if (tokens.Count == 0 || currentToken.Length > 0)
        {
            tokens.Add(_vocabulary["[UNK]"]);
        }

        return tokens;
    }
}

