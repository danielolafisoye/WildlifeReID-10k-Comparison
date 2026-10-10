function pct(score) {
  return `${(score * 100).toFixed(1)}%`;
}

export default function ResultPanel({ result }) {
  const known = !result.is_unknown;

  return (
    <section className={`card result ${known ? "known" : "unknown"}`} aria-live="polite">
      <h2>{known ? "Known individual" : "Unknown individual"}</h2>
      <p className="headline">
        {known ? result.predicted_identity : "No confident known identity was found."}
      </p>
      <p className="muted">
        Best guess: <strong>{result.predicted_identity}</strong> · {pct(result.confidence)} confidence
      </p>

      <h3>Top {result.top_5_details.length} predictions</h3>
      <ol className="candidates">
        {result.top_5_details.map((candidate) => (
          <li key={candidate.identity}>
            <span>{candidate.identity}</span>
            <span className="bar" style={{ "--w": `${candidate.confidence * 100}%` }} />
            <span className="score">{pct(candidate.confidence)}</span>
          </li>
        ))}
      </ol>

      <p className="muted small">Model: {result.model} · {result.inference_ms} ms</p>
    </section>
  );
}
