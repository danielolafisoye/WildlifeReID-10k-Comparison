import { useEffect, useState } from "react";
import { apiAssetUrl, getTestImages } from "../api.js";

export default function TestImagePicker({ selected, onSelect, disabled }) {
  const [page, setPage] = useState(1);
  const [type, setType] = useState("all");
  const [data, setData] = useState(null);
  const [error, setError] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    let active = true;
    setLoading(true);
    setError(null);
    getTestImages({ page, type })
      .then((response) => active && setData(response))
      .catch((requestError) => active && setError(requestError.message))
      .finally(() => active && setLoading(false));
    return () => {
      active = false;
    };
  }, [page, type]);

  function changeType(nextType) {
    setType(nextType);
    setPage(1);
    onSelect(null);
  }

  if (error) {
    return <p className="inline-message">Test images are unavailable: {error}</p>;
  }

  return (
    <div className="test-browser">
      <div className="filter-row" aria-label="Test image filters">
        {["all", "known", "unknown"].map((option) => (
          <button
            type="button"
            key={option}
            className={type === option ? "filter active" : "filter"}
            onClick={() => changeType(option)}
            disabled={disabled}
          >
            {option}
          </button>
        ))}
      </div>

      {loading ? (
        <div className="picker-loading">Loading test images…</div>
      ) : (
        <div className="image-grid">
          {data?.items.map((item) => (
            <button
              type="button"
              className={selected?.id === item.id ? "image-choice selected" : "image-choice"}
              key={item.id}
              onClick={() => onSelect(item)}
              disabled={disabled}
              aria-label={`Choose ${item.query_type} test image`}
            >
              <img src={apiAssetUrl(item.image_url)} alt="Wildlife test query" loading="lazy" />
              <span className={`query-badge ${item.query_type}`}>{item.query_type}</span>
            </button>
          ))}
        </div>
      )}

      {data && data.pages > 1 && (
        <div className="pagination">
          <button type="button" onClick={() => setPage((value) => value - 1)} disabled={disabled || page <= 1}>
            Previous
          </button>
          <span>Page {page} of {data.pages}</span>
          <button
            type="button"
            onClick={() => setPage((value) => value + 1)}
            disabled={disabled || page >= data.pages}
          >
            Next
          </button>
        </div>
      )}
    </div>
  );
}
