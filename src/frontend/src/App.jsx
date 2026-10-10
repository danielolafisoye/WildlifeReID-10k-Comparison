import { useEffect, useState } from "react";
import { getConfig, getHealth, predict } from "./api.js";
import UploadDropzone from "./components/UploadDropzone.jsx";
import ResultPanel from "./components/ResultPanel.jsx";

export default function App() {
  const [health, setHealth] = useState(null);
  const [config, setConfig] = useState(null);
  const [file, setFile] = useState(null);
  const [previewUrl, setPreviewUrl] = useState(null);
  const [threshold, setThreshold] = useState(null);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  useEffect(() => {
    Promise.all([getHealth(), getConfig()])
      .then(([h, c]) => {
        setHealth(h);
        setConfig(c);
        setThreshold(c.threshold);
      })
      .catch(() => setError("Cannot reach the API. Is the Flask server running on port 5000?"));
  }, []);

  useEffect(() => {
    if (!file) return undefined;
    const url = URL.createObjectURL(file);
    setPreviewUrl(url);
    return () => URL.revokeObjectURL(url);
  }, [file]);

  function onSelect(selected) {
    setFile(selected);
    setResult(null);
    setError(null);
  }

  async function onSubmit() {
    if (!file) return;
    setLoading(true);
    setError(null);
    try {
      setResult(await predict(file, threshold));
    } catch (err) {
      setResult(null);
      setError(err.message);
    } finally {
      setLoading(false);
    }
  }

  return (
    <main className="app">
      <header>
        <h1>Wildlife Re-Identification</h1>
        <p className="subtitle">
          Upload an animal photo to find the matching known individual, or learn that it has not been seen before.
        </p>
        {health && (
          <p className="meta">
            Model: <strong>{health.model}</strong> · Gallery: {health.gallery_identities} individuals,{" "}
            {health.gallery_images} images
          </p>
        )}
      </header>

      <section className="card">
        <UploadDropzone onSelect={onSelect} previewUrl={previewUrl} disabled={loading} config={config} />

        {threshold !== null && (
          <label className="threshold">
            Unknown-individual threshold: <strong>{Number(threshold).toFixed(2)}</strong>
            <input
              type="range"
              min="0"
              max="1"
              step="0.01"
              value={threshold}
              onChange={(e) => setThreshold(Number(e.target.value))}
              disabled={loading}
            />
          </label>
        )}

        <button className="primary" onClick={onSubmit} disabled={!file || loading}>
          {loading ? "Identifying…" : "Identify animal"}
        </button>

        {error && (
          <p className="error" role="alert">
            {error}
          </p>
        )}
      </section>

      {result && <ResultPanel result={result} />}
    </main>
  );
}
