namespace EmbeddingsApi2
{
    using System;
    using System.Collections.Generic;
    using System.Linq;
    using System.Text;
    using System.Threading.Tasks;

    /// <summary>
    /// GGUF file details.
    /// </summary>
    public class GgufFileDetails
    {
        #region Public-Members

        /// <summary>
        /// HuggingFace BLOB identifier.
        /// </summary>
        public string ObjectIdentifier { get; set; } = null;

        /// <summary>
        /// Filename.
        /// </summary>
        public string Filename { get; set; } = null;

        /// <summary>
        /// Content length.
        /// </summary>
        public long ContentLength
        {
            get
            {
                return _ContentLength;  
            }
            set
            {
                if (value < 0) throw new ArgumentOutOfRangeException(nameof(ContentLength));
                _ContentLength = value;
            }
        }

        #endregion

        #region Private-Members

        private long _ContentLength = 0;

        #endregion

        #region Constructors-and-Factories

        /// <summary>
        /// GGUF file details.
        /// </summary>
        public GgufFileDetails()
        {

        }

        #endregion

        #region Public-Methods

        #endregion

        #region Private-Methods

        #endregion
    }
}