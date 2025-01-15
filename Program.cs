// using Microsoft.AspNetCore.Builder;
// using Microsoft.AspNetCore.Hosting;
// using Microsoft.Extensions.Hosting;
// using Microsoft.AspNetCore.Http;
// using System.Text.Json;
// using System.Text.Json.Serialization;
// using System.Threading;
// using System.Threading.Tasks;
// using EmbeddingsApi2;
// using Microsoft.ML.OnnxRuntime;
// using Microsoft.ML.OnnxRuntime.Tensors;
//
// // LLamaSharp
// using LLama;
// using LLama.Common;
//
// // OnnxRuntime (needed if you go the ONNX route for MiniLM)
// using Microsoft.ML.OnnxRuntime;
//
// // If you have a reference to your HuggingFaceClient, SyslogLogging, etc:
// var builder = WebApplication.CreateBuilder(args);
//
// var app = builder.Build();
//
// // 1) GET /health – to check if it is running
// app.MapGet("/health", () => 
// {
//     return Results.Ok("Service is running!");
// });
//
// app.MapPost("/preload", async (PreloadRequest req) =>
// {
//     // Logging + serializer for your HuggingFaceClient
//     var logging = new SyslogLogging.LoggingModule();
//     var serializer = new Serializer();
//
//     var huggingFaceClient = new HuggingFaceClient(logging, serializer, req.ApiKey ?? "");
//
//     // Step A: If requested, download the model from Hugging Face
//     //         (e.g., for LLaMA .gguf or an ONNX file for MiniLM).
//     try
//     {
//         if (!string.IsNullOrEmpty(req.ModelId))
//         {
//             Console.WriteLine($"[preload] Downloading model files for '{req.ModelId}'...");
//
//             // If it's LLaMA-based .gguf
//             if (req.ModelFormat?.Equals("gguf", StringComparison.OrdinalIgnoreCase) == true)
//             {
//                 // We'll call HuggingFaceClient.DownloadGguf
//                 bool success = await huggingFaceClient.DownloadGguf(
//                     req.ModelId,
//                     req.SourceFilename!,   // e.g. "mymodel.gguf"
//                     req.OutputFilename!,   // e.g. "mymodel.gguf"
//                     CancellationToken.None
//                 );
//
//                 if (!success)
//                 {
//                     Console.WriteLine($"[preload] Failed to download '{req.SourceFilename}'.");
//                     return Results.BadRequest("Failed to download model file.");
//                 }
//                 Console.WriteLine($"[preload] Downloaded '{req.SourceFilename}' to '{req.OutputFilename}'.");
//             }
//             else
//             {
//                 // Maybe we have a non-LLaMA model that we want as an ONNX file
//                 // If you already have a local ONNX, you might skip downloading.
//                 // Or you have some custom approach: e.g. "DownloadOnnx(...)"
//                 // For demonstration, let's just assume you have it locally
//                 Console.WriteLine("[preload] Skipping direct download (ONNX or other).");
//             }
//         }
//     }
//     catch (Exception ex)
//     {
//         Console.WriteLine($"[preload] Error downloading model: {ex.Message}");
//         return Results.BadRequest($"Error: {ex.Message}");
//     }
//
//     // Step B: Load the model into memory for embeddings
//     try
//     {
//         IEmbedder embedder;
//         if (req.ModelFormat?.Equals("gguf", StringComparison.OrdinalIgnoreCase) == true)
//         {
//             // (1) LLaMA-based approach with LLamaSharp
//             Console.WriteLine("[preload] Loading GGUF model with LLamaSharp...");
//             var mParams = new ModelParams(req.OutputFilename!)
//             {
//                 EmbeddingMode = true
//             };
//
//             var weights = LLamaWeights.LoadFromFile(mParams);
//             embedder = new LlamaLocalEmbedder(new LLamaEmbedder(weights, mParams));
//         }
//         else
//         {
//             // (2) MiniLM (or any other) approach using ONNX (skeleton code)
//             Console.WriteLine("[preload] Loading ONNX-based model for MiniLM or similar...");
//             // e.g. "all-minilm-l6-v2.onnx" previously downloaded
//
//             // This is a placeholder for local inference:
//             // Actual usage requires tokenization + session:
//             var miniLmEmbedder = new MiniLmLocalEmbedder("my-minilm-l6-v2.onnx");
//             embedder = miniLmEmbedder;
//         }
//
//         // Store in ModelCache
//         ModelCache.Embedder = embedder;
//         Console.WriteLine("[preload] Model is preloaded successfully.");
//
//         return Results.Ok("Model preloaded successfully!");
//     }
//     catch (Exception ex)
//     {
//         Console.WriteLine($"[preload] Error while loading model: {ex.Message}");
//         return Results.BadRequest($"Error loading model: {ex.Message}");
//     }
// });
//
// // 3) POST /embeddings – to generate embeddings
// //    Expects a JSON body with { "Texts": ["some text", "another text"] }
// app.MapPost("/embeddings", async (EmbeddingRequest req) =>
// {
//     if (!ModelCache.IsModelLoaded)
//     {
//         return Results.BadRequest("Model is not loaded. Call /preload first.");
//     }
//
//     try
//     {
//         var embedder = ModelCache.Embedder!; // non-null because IsModelLoaded is true
//
//         // We'll generate embeddings for each text in the request
//         var embeddingsList = new List<float[]>();
//
//         foreach (var text in req.Texts)
//         {
//             float[] emb = await embedder.GetEmbeddingsAsync(text);
//             embeddingsList.Add(emb);
//         }
//         return Results.Ok(new EmbeddingResponse(embeddingsList));
//     }
//     catch (Exception ex)
//     {
//         Console.WriteLine($"[embeddings] Error while getting embeddings: {ex.Message}");
//         return Results.BadRequest($"Error while generating embeddings: {ex.Message}");
//     }
// });
//
// // Run the app
// app.Run();
//
// // For demonstration, you can define them in the same file or as separate classes.
// record EmbeddingRequest(List<string> Texts);
// record EmbeddingResponse(List<float[]> Embeddings);
//
// // A static class to hold the embedder once loaded
// static class ModelCache
// {
//     public static IEmbedder? Embedder { get; set; }
//     public static bool IsModelLoaded => Embedder != null;
// }
//
// // Minimal API Program
//
//
// // 2) POST /preload – to download and load the model
// //    This example uses your HuggingFaceClient logic from your snippet.
//


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

// 2) POST /preload – Load either a GGUF or ONNX model
app.MapPost("/preload", async (PreloadRequest req) =>
{
    try
    {
        Console.WriteLine($"[preload] Received request to load model: {req.ModelId}");

        // Step 1: Load the appropriate model type
        IEmbedder embedder;
        if (req.ModelFormat?.Equals("gguf", StringComparison.OrdinalIgnoreCase) == true)
        {
            // GGUF (LLaMA-based models)
            Console.WriteLine("[preload] Loading GGUF model with LLamaSharp...");
            var mParams = new ModelParams(req.OutputFilename!)
            {
                EmbeddingMode = true
            };
            var weights = LLamaWeights.LoadFromFile(mParams);
            embedder = new LlamaLocalEmbedder(new LLamaEmbedder(weights, mParams));
        }
        else if (req.ModelFormat?.Equals("onnx", StringComparison.OrdinalIgnoreCase) == true)
        {
            // ONNX (MiniLM or similar)
            Console.WriteLine("[preload] Loading ONNX model...");
    
            // Initialize HuggingFaceClient
            var logging = new SyslogLogging.LoggingModule();
            var serializer = new Serializer();
            var huggingFaceClient = new HuggingFaceClient(logging, serializer, req.ApiKey ?? "");

            // Download ONNX model file if needed
            string onnxFilename = "model.onnx";  // The filename in the Hugging Face repo
            string localOnnxPath = Path.Combine(Directory.GetCurrentDirectory(), req.OutputFilename ?? "model.onnx");
    
            if (!File.Exists(localOnnxPath))
            {
                Console.WriteLine("[preload] Downloading ONNX model file...");
                bool success = await huggingFaceClient.DownloadOnnx(
                    "onnx-models/all-MiniLM-L6-v2-onnx",
                    onnxFilename,
                    localOnnxPath,
                    CancellationToken.None
                );

                if (!success)
                {
                    return Results.BadRequest("Failed to download ONNX model file.");
                }
            }

            // Download vocabulary file if needed
            string vocabFilename = "vocab.txt";
            string localVocabPath = Path.Combine(Directory.GetCurrentDirectory(), vocabFilename);
    
            if (!File.Exists(localVocabPath))
            {
                Console.WriteLine("[preload] Downloading vocabulary file...");
                bool success = await huggingFaceClient.DownloadVocabulary(
                    "onnx-models/all-MiniLM-L6-v2-onnx",
                    vocabFilename,
                    localVocabPath,
                    CancellationToken.None
                );

                if (!success)
                {
                    return Results.BadRequest("Failed to download vocabulary file.");
                }
            }

            embedder = new MiniLmLocalEmbedder(localOnnxPath, localVocabPath);
        }
        else
        {
            return Results.BadRequest("Invalid model format. Supported formats: gguf, onnx.");
        }

        // Step 2: Store embedder in ModelCache
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
record PreloadRequest(string? ModelId, string? ModelFormat, string? OutputFilename, string? ApiKey);
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

