import { useState } from "react";
import "./App.css";
import DistributionCards from "./components/distribution_cards.jsx";

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
      const apiUrl = import.meta.env.VITE_API_URL || "http://localhost:5000";
      const response = await fetch(`${apiUrl}/upload`, {
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
    <div className="min-h-screen w-screen flex flex-col justify-center items-center bg-[#0f0f13] text-[#eeeeee] font-sans">
      <div className="max-w-[1200px] mx-auto p-8 text-center">
        <header>
          <h1 className="text-6xl mb-2 font-extrabold bg-gradient-to-tr from-[#4ecca3] to-[#45a29e] bg-clip-text text-transparent">
            COLOR ANALYZER
          </h1>
          <p className="text-[#a6a6a6] text-xl mb-12">
            Upload the image to analyze the colors
          </p>
        </header>

        <main>
          <div className="mb-16 flex flex-col items-center gap-8">
            <input
              type="file"
              accept="image/*"
              onChange={handleFileChange}
              id="file-upload"
              className="hidden"
            />
            <label
              htmlFor="file-upload"
              className="bg-[#1a1a2e] px-10 py-4 rounded-full cursor-pointer border-2 border-[#4ecca3] text-[#4ecca3] transition-all duration-300 font-bold text-lg hover:bg-[#4ecca3] hover:text-[#0f0f13] hover:-translate-y-0.5 hover:shadow-[0_5px_15px_rgba(78,204,163,0.3)]"
            >
              {selectedFile ? "Change Image" : "Select Image"}
            </label>

            {preview && (
              <div className="flex flex-col items-center gap-6 w-full animate-[fadeIn_0.5s_ease]">
                <img
                  src={preview}
                  alt="Preview"
                  className="max-w-full max-h-[400px] rounded-xl shadow-[0_10px_30px_rgba(0,0,0,0.5)] border border-white/10"
                />
                <button
                  onClick={handleUpload}
                  className="bg-gradient-to-tr from-[#4ecca3] to-[#45a29e] border-none px-12 py-4 rounded-full text-[#0f0f13] text-lg font-bold cursor-pointer transition-all duration-300 hover:-translate-y-0.5 hover:shadow-[0_5px_20px_rgba(78,204,163,0.4)] disabled:opacity-70 disabled:cursor-not-allowed"
                  disabled={loading}
                >
                  {loading ? "Analyzing..." : "Analyze"}
                </button>
              </div>
            )}
          </div>

          {results && (
            <div className="grid grid-cols-[repeat(auto-fit,minmax(300px,1fr))] gap-8 animate-[slideUp_0.8s_ease]">
              <DistributionCards
                title="Color Distribution"
                image={results.color_distribution}
                alt="Color Distribution"
              />
              <DistributionCards
                title="Hue Wheel"
                image={results.hue_distribution}
                alt="Hue Wheel"
              />
              <DistributionCards
                title="Brightness"
                image={results.brightness_distribution}
                alt="Brightness"
              />
              <DistributionCards
                title="Saturation"
                image={results.saturation_distribution}
                alt="Saturation"
              />
              <DistributionCards
                title="3D Distribution"
                image={results.histogram_3d}
                alt="3D Distribution"
              />
            </div>
          )}
        </main>
      </div>

      <div>
        <p className="font-semibold text-[#a6a6a6] text-xl my-10">
          This site has been created by Kyoronginus
        </p>
      </div>
    </div>
  );
}

export default App;
