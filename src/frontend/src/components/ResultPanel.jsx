function pct(score) {
  return `${(score * 100).toFixed(1)}%`;
}

export default function ResultPanel({ result }) {
  const known = result.status === "known";
  const emptyGallery = result.reason === "gallery_empty";

  return (
    <section className={`card result ${known ? "known" : "unknown"}`} aria-live="polite">
      {known ? (
        <>
          <h2>Known individual</h2>
          <p className="headline">
            {result.prediction.identity} <span className="score">{pct(result.prediction.score)} similarity</span>
          </p>
          {result.score === "margin" && (
            <p className="muted">
              {pct(result.decision_score)} ahead of the second-best individual (at least {pct(result.threshold)} required).
            </p>
          )}
        </>
      ) : (
        <>
          <h2>Unknown individual</h2>
          <p className="headline">
            {emptyGallery
              ? "The gallery is empty, so there is nothing to compare against. Build the gallery first."
              : "No sufficiently reliable known identity was found."}
          </p>
          {result.prediction.score !== null && result.score === "margin" && (
            <p className="muted">
              The closest match ({pct(result.prediction.score)} similarity) was only {pct(result.decision_score)} ahead
              of the second-best individual; a lead of at least {pct(result.threshold)} is required.
            </p>
          )}
          {result.prediction.score !== null && result.score !== "margin" && (
            <p className="muted">
              Best similarity was {pct(result.prediction.score)}, below the threshold of {pct(result.threshold)}.
            </p>
          )}
        </>
      )}

      {result.top_k.length > 0 && (
        <>
          <h3>Top {result.top_k.length} candidates</h3>
          <ol className="candidates">
            {result.top_k.map((c) => (
              <li key={c.identity}>
                <span>{c.identity}</span>
                <span className="bar" style={{ "--w": `${Math.max(0, Math.min(1, c.score)) * 100}%` }} />
                <span className="score">{pct(c.score)}</span>
              </li>
            ))}
          </ol>
        </>
      )}

      <p className="muted small">
        Model: {result.model} · {result.inference_ms} ms
      </p>
    </section>
  );
}
