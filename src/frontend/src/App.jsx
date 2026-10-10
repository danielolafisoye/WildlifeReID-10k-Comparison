import { useEffect, useState } from "react";
import { apiAssetUrl, getConfig, getHealth, predict } from "./api.js";
import UploadDropzone from "./components/UploadDropzone.jsx";
import ResultPanel from "./components/ResultPanel.jsx";
import TestImagePicker from "./components/TestImagePicker.jsx";

export default function App() {
  const [health, setHealth] = useState(null);
  const [config, setConfig] = useState(null);
  const [model, setModel] = useState("convnext");
  const [inputMode, setInputMode] = useState("test");
  const [file, setFile] = useState(null);
  const [selectedTest, setSelectedTest] = useState(null);
  const [previewUrl, setPreviewUrl] = useState(null);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  useEffect(() => {
    Promise.all([getHealth(), getConfig()])
      .then(([healthResponse, configResponse]) => {
        setHealth(healthResponse);
        setConfig(configResponse);
        setModel(configResponse.default_model);
      })
      .catch(() => setError("Cannot reach the API. Is the Flask server running on port 5000?"));
  }, []);

  useEffect(() => {
    if (!file) {
      setPreviewUrl(null);
      return undefined;
    }
    const url = URL.createObjectURL(file);
    setPreviewUrl(url);
    return () => URL.revokeObjectURL(url);
  }, [file]);

  function selectMode(nextMode) {
    setInputMode(nextMode);
    setFile(null);
    setSelectedTest(null);
    setResult(null);
    setError(null);
  }

  function selectFile(selected) {
    setFile(selected);
    setSelectedTest(null);
    setResult(null);
    setError(null);
  }

  function selectTestImage(selected) {
    setSelectedTest(selected);
    setFile(null);
    setResult(null);
    setError(null);
  }

  async function identify() {
    if (!file && !selectedTest) return;
    setLoading(true);
    setError(null);
    try {
      setResult(await predict({ model, file, testImageId: selectedTest?.id }));
    } catch (requestError) {
      setResult(null);
      setError(requestError.message);
    } finally {
      setLoading(false);
    }
  }

  const selectedPreview = selectedTest ? apiAssetUrl(selectedTest.image_url) : previewUrl;

  return (
    <main className="app">
      <header>
        <h1>Wildlife Re-Identification</h1>
        <p className="subtitle">
          Choose a test image or upload an animal photo to identify a known individual, or detect an unknown one.
        </p>
        {health && (
          <p className="meta">
            API connected · Unknown threshold: <strong>{Math.round(config.threshold * 100)}%</strong>
          </p>
        )}
      </header>

      <section className="card">
        <label className="field">
          <strong>Model</strong>
          <select value={model} onChange={(event) => setModel(event.target.value)} disabled={loading}>
            {(config?.models ?? []).map((option) => (
              <option key={option.id} value={option.id} disabled={!option.available}>
                {option.name}{option.available ? "" : " (not available)"}
              </option>
            ))}
          </select>
        </label>

        <div className="mode-tabs">
          <button type="button" className={inputMode === "test" ? "active" : ""} onClick={() => selectMode("test")}>Test images</button>
          <button type="button" className={inputMode === "upload" ? "active" : ""} onClick={() => selectMode("upload")}>Upload image</button>
        </div>

        {inputMode === "test" ? (
          <TestImagePicker selected={selectedTest} onSelect={selectTestImage} disabled={loading} />
        ) : (
          <UploadDropzone onSelect={selectFile} previewUrl={previewUrl} disabled={loading} config={config} />
        )}

        {selectedTest && (
          <div className="selected-image">
            <img src={selectedPreview} alt="Selected test query" />
            <span>Test image selected</span>
          </div>
        )}

        <button className="primary" type="button" onClick={identify} disabled={(!file && !selectedTest) || loading}>
          {loading ? "Identifying…" : "Identify animal"}
        </button>

        {error && <p className="error" role="alert">{error}</p>}
      </section>

      {result && <ResultPanel result={result} />}
    </main>
  );
}
