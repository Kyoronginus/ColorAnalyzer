import { useState } from "react";
import "./App.css";

function App() {
  const [selectedFile, setSelectedFile] = useState(null);
  const [preview, setPreview] = useState(null);
  const [results, setResults] = useState(null);
  const [loading, setLoading] = useState(false);

  const handleFileChange = (e) => {
    const file = e.target.files[0];
    if (file) {
      setSelectedFile(file);
      setPreview(URL.createObjectURL(file));
      setResults(null);
    }
  };

  const handleUpload = async () => {
    if (!selectedFile) return;

    setLoading(true);
    const formData = new FormData();
    formData.append("image", selectedFile);

    try {
      // Flask port 5000
      const response = await fetch("http://localhost:5000/upload", {
        method: "POST",
        body: formData,
      });
      const data = await response.json();
      if (response.ok) {
        setResults(data);
      } else {
        console.error("Upload failed:", data.error);
        alert("Upload failed: " + data.error);
      }
    } catch (error) {
      console.error("Error:", error);
      alert("Error connecting to server");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="container">
      <header>
        <h1>COLOR ANALYZER</h1>
        <p>Upload the image to analyze the colors</p>
      </header>

      <main>
        <div className="upload-section">
          <input
            type="file"
            accept="image/*"
            onChange={handleFileChange}
            id="file-upload"
            className="file-input"
          />
          <label htmlFor="file-upload" className="file-label">
            {selectedFile ? "Change Image" : "Select Image"}
          </label>

          {preview && (
            <div className="preview-container">
              <img src={preview} alt="Preview" className="preview-image" />
              <button
                onClick={handleUpload}
                className="analyze-button"
                disabled={loading}
              >
                {loading ? "Analyzing..." : "Analyze Colors"}
              </button>
            </div>
          )}
        </div>

        {results && (
          <div className="results-grid">
            {/* <div className="result-card">
              <h3>Simplified Colors</h3>
              <img src={results.simplified_image} alt="Simplified" />
            </div> */}
            <div className="result-card">
              <h3>Color Distribution</h3>
              <img src={results.color_distribution} alt="Color Distribution" />
            </div>
            <div className="result-card">
              <h3>Hue Wheel</h3>
              <img src={results.hue_distribution} alt="Hue Wheel" />
            </div>
            <div className="result-card">
              <h3>Brightness</h3>
              <img src={results.brightness_distribution} alt="Brightness" />
            </div>
            <div className="result-card">
              <h3>Saturation</h3>
              <img src={results.saturation_distribution} alt="Saturation" />
            </div>
            <div className="result-card">
              <h3>3D Distribution</h3>
              <img src={results.histogram_3d} alt="3D Histogram" />
            </div>
          </div>
        )}
      </main>
    </div>
  );
}

export default App;
