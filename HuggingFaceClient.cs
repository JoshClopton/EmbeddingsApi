namespace EmbeddingsApi2
{
    using System;
    using System.Collections.Generic;
    using System.IO;
    using System.Net.Http;
    using System.Text.Json;
    using System.Threading;
    using System.Threading.Tasks;
    using RestWrapper;
    using SyslogLogging;

    /// <summary>
    /// HuggingFace client.
    /// </summary>
    public class HuggingFaceClient
    {
        #region Public-Members

        /// <summary>
        /// Endpoint URL of the form [protocol]://[hostname]/.  
        /// Default is https://huggingface.co/api/models/.
        /// </summary>
        public string Endpoint
        {
            get
            {
                return _Endpoint;
            }
            set
            {
                if (String.IsNullOrEmpty(value)) throw new ArgumentNullException(nameof(Endpoint));
                Uri uri = new Uri(value); // test
                if (!value.EndsWith("/")) value += "/";
                _Endpoint = value;
            }
        }

        /// <summary>
        /// Stream buffer size.
        /// </summary>
        public int StreamBufferSize
        {
            get
            {
                return _StreamBufferSize;
            }
            set
            {
                if (value < 1) throw new ArgumentOutOfRangeException(nameof(StreamBufferSize));
                _StreamBufferSize = value;
            }
        }

        #endregion

        #region Private-Members

        private string _Header = "[HuggingFaceClient] ";
        private LoggingModule _Logging = null;
        private Serializer _Serializer = null;
        private string _ApiKey = null;
        private string _Endpoint = "https://huggingface.co/";
        private int _StreamBufferSize = 8192;

        #endregion

        #region Constructors-and-Factories

        /// <summary>
        /// HuggingFace client.
        /// </summary>
        /// <param name="logging">Logging.</param>
        /// <param name="serializer">Serializer.</param>
        /// <param name="apiKey">HuggingFace API key.</param>
        /// <param name="endpoint">Endpoint URL of the form [protocol]://[hostname]/.  Default is https://huggingface.co/.</param>
        public HuggingFaceClient(LoggingModule logging, Serializer serializer, string apiKey, string endpoint = "https://huggingface.co/")
        {
            _Logging = logging ?? throw new ArgumentNullException(nameof(logging));
            _Serializer = serializer ?? throw new ArgumentNullException(nameof(serializer));
            _ApiKey = apiKey;

            Endpoint = endpoint;
        }

        #endregion

        #region Public-Methods

        /// <summary>
        /// List available GGUF files for a given model.
        /// </summary>
        /// <param name="modelId">Fully qualified model ID, including the owner name.  Input must be of the form {owner}/{model}.</param>
        /// <param name="token">Cancellation token.</param>
        /// <returns>List of GGUF filenames.</returns>
        public async Task<List<GgufFileDetails>> ListAvailableGgufFiles(string modelId, CancellationToken token = default)
        {
            if (String.IsNullOrEmpty(modelId)) throw new ArgumentNullException(nameof(modelId));

            // https://huggingface.co/api/models/{owner}/{model}/tree/main
            string url = _Endpoint + "api/models/" + modelId + "/tree/main";

            using (RestRequest req = new RestRequest(url))
            {
                if (!String.IsNullOrEmpty(_ApiKey)) req.Authorization.BearerToken = _ApiKey;

                using (RestResponse resp = await req.SendAsync(token).ConfigureAwait(false))
                {
                    if (resp == null)
                    {
                        _Logging.Warn(_Header + "no response from " + url);
                        return null;
                    }
                    else
                    {
                        if (resp.StatusCode >= 200 && resp.StatusCode <= 299)
                        {
                            _Logging.Debug(_Header + "success response from " + url);

                            if (!String.IsNullOrEmpty(resp.DataAsString))
                            {
                                _Logging.Debug(_Header + "deserializing response body");

                                List<Dictionary<string, object>> listDict = _Serializer.DeserializeJson<List<Dictionary<string, object>>>(resp.DataAsString);
                                if (listDict == null || listDict.Count < 1)
                                {
                                    _Logging.Warn(_Header + "null or empty dictionary returned from " + url + " for " + modelId);
                                    return null;
                                }

                                List<GgufFileDetails> ret = new List<GgufFileDetails>();

                                foreach (Dictionary<string, object> dict in listDict)
                                {
                                    string path = dict.GetValueOrDefault("path")?.ToString();
                                    if (String.IsNullOrEmpty(path) || !path.EndsWith(".gguf")) continue;

                                    string oid = dict.GetValueOrDefault("oid")?.ToString();
                                    long contentLength = dict.GetValueOrDefault("size") is long l ? l : 0;

                                    ret.Add(new GgufFileDetails
                                    {
                                        Filename = path,
                                        ContentLength = contentLength,
                                        ObjectIdentifier = oid
                                    });
                                }

                                return ret;
                            }
                            else
                            {
                                _Logging.Warn(_Header + "no response body from " + url);
                                return null;
                            }
                        }
                        else
                        {
                            _Logging.Warn(_Header + "failure response from " + url + ": " + resp.StatusCode);
                            return null;
                        }
                    }
                }
            }
        }

        /// <summary>
        /// Download GGUF model file.
        /// </summary>
        /// <param name="modelId">Fully qualified model ID, including the owner name.  Input must be of the form {owner}/{model}.</param>
        /// <param name="ggufFilename">GGUF filename.</param>
        /// <param name="outputFilename">Output filename.</param>
        /// <param name="token">Cancellation token.</param>
        /// <returns>True if successful.</returns>
        public async Task<bool> DownloadGguf(string modelId, string ggufFilename, string outputFilename, CancellationToken token = default)
        {
            if (string.IsNullOrEmpty(modelId)) throw new ArgumentNullException(nameof(modelId));
            if (string.IsNullOrEmpty(ggufFilename)) throw new ArgumentNullException(nameof(ggufFilename));
            if (string.IsNullOrEmpty(outputFilename)) throw new ArgumentNullException(nameof(outputFilename));

            string url = _Endpoint + modelId + "/resolve/main/" + ggufFilename;

            try
            {
                using (RestRequest req = new RestRequest(url))
                {
                    if (!string.IsNullOrEmpty(_ApiKey))
                        req.Authorization.BearerToken = _ApiKey;

                    using (RestResponse resp = await req.SendAsync(token).ConfigureAwait(false))
                    {
                        if (resp == null)
                        {
                            _Logging.Warn($"{_Header}no response from {url}");
                            return false;
                        }

                        if (resp.StatusCode >= 200 && resp.StatusCode <= 299)
                        {
                            _Logging.Debug($"{_Header}success response from {url}");

                            if (resp.ContentLength != null && resp.ContentLength.Value > 0)
                            {
                                _Logging.Debug($"{_Header}Receiving {resp.ContentLength.Value} bytes.");

                                byte[] buffer = new byte[_StreamBufferSize];
                                long totalBytes = resp.ContentLength.Value;
                                long remaining = totalBytes;
                                long writtenSoFar = 0;

                                try
                                {
                                    using (FileStream fs = new FileStream(outputFilename, FileMode.Create, FileAccess.Write))
                                    {
                                        while (remaining > 0)
                                        {
                                            int read = await resp.Data.ReadAsync(buffer, 0, buffer.Length).ConfigureAwait(false);
                                            if (read > 0)
                                            {
                                                await fs.WriteAsync(buffer, 0, read).ConfigureAwait(false);
                                                remaining -= read;
                                                writtenSoFar += read;

                                                // Update the terminal line with progress
                                                double progress = ((double)(totalBytes - remaining) / totalBytes) * 100;
                                                Console.Write($"\rProgress: {progress:F2}% ({writtenSoFar}/{totalBytes} bytes)");
                                            }
                                            else
                                            {
                                                _Logging.Warn($"{_Header}No more data to read from the stream.");
                                                break;
                                            }
                                        }

                                        Console.WriteLine(); // End the progress line
                                        await fs.FlushAsync().ConfigureAwait(false);
                                        _Logging.Debug($"{_Header}File writing completed and flushed.");
                                    }

                                    _Logging.Debug($"{_Header}Successfully downloaded {resp.ContentLength.Value} bytes to file '{outputFilename}'.");
                                    return true;
                                }
                                catch (Exception ex)
                                {
                                    _Logging.Error($"{_Header}Error while writing to file: {ex.Message}");
                                    return false;
                                }
                            }
                            else
                            {
                                _Logging.Warn($"{_Header}Response body is empty or content length is zero.");
                                return false;
                            }
                        }
                        else
                        {
                            _Logging.Warn($"{_Header}Failure response from {url}: {resp.StatusCode}");
                            return false;
                        }
                    }
                }
            }
            catch (Exception ex)
            {
                _Logging.Error($"{_Header}Error during download: {ex.Message}");
                return false;
            }
        }
        
        #endregion
        
        /// <summary>
        /// Download vocabulary file.
        /// </summary>
        /// <param name="modelId">Fully qualified model ID, including the owner name.  Input must be of the form {owner}/{model}.</param>
        /// <param name="vocabFilename">Vocabulary filename (e.g., 'vocab.txt').</param>
        /// <param name="outputFilename">Output filename.</param>
        /// <param name="token">Cancellation token.</param>
        /// <returns>True if successful.</returns>
        public async Task<bool> DownloadVocabulary(string modelId, string vocabFilename, string outputFilename, CancellationToken token = default)
        {
            if (string.IsNullOrEmpty(modelId)) throw new ArgumentNullException(nameof(modelId));
            if (string.IsNullOrEmpty(vocabFilename)) throw new ArgumentNullException(nameof(vocabFilename));
            if (string.IsNullOrEmpty(outputFilename)) throw new ArgumentNullException(nameof(outputFilename));

            string url = _Endpoint + modelId + "/resolve/main/" + vocabFilename;

            try
            {
                using (RestRequest req = new RestRequest(url))
                {
                    if (!string.IsNullOrEmpty(_ApiKey))
                        req.Authorization.BearerToken = _ApiKey;

                    using (RestResponse resp = await req.SendAsync(token).ConfigureAwait(false))
                    {
                        if (resp == null)
                        {
                            _Logging.Warn($"{_Header}no response from {url}");
                            return false;
                        }

                        if (resp.StatusCode >= 200 && resp.StatusCode <= 299)
                        {
                            _Logging.Debug($"{_Header}success response from {url}");

                            if (resp.ContentLength != null && resp.ContentLength.Value > 0)
                            {
                                try
                                {
                                    using (FileStream fs = new FileStream(outputFilename, FileMode.Create, FileAccess.Write))
                                    {
                                        await resp.Data.CopyToAsync(fs);
                                        await fs.FlushAsync().ConfigureAwait(false);
                                    }

                                    _Logging.Debug($"{_Header}Successfully downloaded vocabulary file to '{outputFilename}'.");
                                    return true;
                                }
                                catch (Exception ex)
                                {
                                    _Logging.Error($"{_Header}Error while writing vocabulary file: {ex.Message}");
                                    return false;
                                }
                            }
                            else
                            {
                                _Logging.Warn($"{_Header}Response body is empty or content length is zero.");
                                return false;
                            }
                        }
                        else
                        {
                            _Logging.Warn($"{_Header}Failure response from {url}: {resp.StatusCode}");
                            return false;
                        }
                    }
                }
            }
            catch (Exception ex)
            {
                _Logging.Error($"{_Header}Error during vocabulary download: {ex.Message}");
                return false;
            }
        }
        
        /// <summary>
        /// Download ONNX model file.
        /// </summary>
        /// <param name="modelId">Fully qualified model ID, including the owner name.</param>
        /// <param name="onnxFilename">ONNX filename.</param>
        /// <param name="outputFilename">Output filename.</param>
        /// <param name="token">Cancellation token.</param>
        /// <returns>True if successful.</returns>
        public async Task<bool> DownloadOnnx(string modelId, string onnxFilename, string outputFilename, CancellationToken token = default)
        {
            // This can use the same implementation as DownloadGguf since the process is the same
            return await DownloadGguf(modelId, onnxFilename, outputFilename, token);
        }

        #region Private-Methods

        #endregion
    }
}