using EmbeddingsApi2;
using EmbeddingsApi2.EmbeddingModels;

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
            // ONNX download logic...
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
