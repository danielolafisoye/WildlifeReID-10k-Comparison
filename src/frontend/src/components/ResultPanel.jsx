import { useEffect, useState } from "react";

import { apiAssetUrl } from "../api";


function pct(score) {
  return `${(score * 100).toFixed(1)}%`;
}

function ReferenceImage({ imageUrl, identity, className, onOpen }) {
  const [failed, setFailed] = useState(false);

  if (!imageUrl || failed) return null;
  return (
    <button
      className="reference-image-button"
      type="button"
      aria-label={`Open a larger reference image for ${identity}`}
      onClick={() => onOpen({ imageUrl, identity })}
    >
      <img
        className={className}
        src={apiAssetUrl(imageUrl)}
        alt={`Reference image for ${identity}`}
        loading="lazy"
        onError={() => setFailed(true)}
      />
    </button>
  );
}

export default function ResultPanel({ result }) {
  const known = !result.is_unknown;
  const bestMatch = result.top_5_details[0];
  const [openImage, setOpenImage] = useState(null);

  useEffect(() => {
    if (!openImage) return undefined;

    const closeOnEscape = (event) => {
      if (event.key === "Escape") setOpenImage(null);
    };
    const previousOverflow = document.body.style.overflow;
    document.body.style.overflow = "hidden";
    window.addEventListener("keydown", closeOnEscape);

    return () => {
      document.body.style.overflow = previousOverflow;
      window.removeEventListener("keydown", closeOnEscape);
    };
  }, [openImage]);

  return (
    <>
      <section className={`card result ${known ? "known" : "unknown"}`} aria-live="polite">
        <h2>{known ? "Known individual" : "Unknown individual"}</h2>
        <div className="identity-summary">
          <ReferenceImage
            className="result-reference"
            imageUrl={bestMatch?.image_url}
            identity={result.predicted_identity}
            onOpen={setOpenImage}
          />
          <div>
            <p className="headline">
              {known ? result.predicted_identity : "No confident known identity was found."}
            </p>
            <p className="muted">
              Best guess: <strong>{result.predicted_identity}</strong> · {pct(result.confidence)} confidence
            </p>
          </div>
        </div>

        <h3>Top {result.top_5_details.length} predictions</h3>
        <ol className="candidates">
          {result.top_5_details.map((candidate) => (
            <li key={candidate.identity}>
              <ReferenceImage
                className="candidate-reference"
                imageUrl={candidate.image_url}
                identity={candidate.identity}
                onOpen={setOpenImage}
              />
              <span className="candidate-name">{candidate.identity}</span>
              <span className="bar" style={{ "--w": `${candidate.confidence * 100}%` }} />
              <span className="score">{pct(candidate.confidence)}</span>
            </li>
          ))}
        </ol>

        <p className="muted reference-note">Click an image to enlarge it. Reference images come from the known training identities.</p>

        <p className="muted small">Model: {result.model} · {result.inference_ms} ms</p>
      </section>

      {openImage && (
        <div
          className="reference-lightbox"
          role="dialog"
          aria-modal="true"
          aria-label={`Reference image for ${openImage.identity}`}
          onMouseDown={(event) => {
            if (event.target === event.currentTarget) setOpenImage(null);
          }}
        >
          <div className="reference-dialog">
            <button
              className="reference-close"
              type="button"
              aria-label="Close image"
              onClick={() => setOpenImage(null)}
              autoFocus
            >
              ×
            </button>
            <img
              className="reference-large"
              src={apiAssetUrl(openImage.imageUrl)}
              alt={`Large reference view for ${openImage.identity}`}
            />
            <p>{openImage.identity}</p>
          </div>
        </div>
      )}
    </>
  );
}
