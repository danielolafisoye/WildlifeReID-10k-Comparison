import { useRef, useState } from "react";

export default function UploadDropzone({ onSelect, previewUrl, disabled, config }) {
  const inputRef = useRef(null);
  const [dragging, setDragging] = useState(false);
  const accept = (config?.allowed_extensions ?? ["jpg", "jpeg", "png", "webp"]).map((e) => `.${e}`).join(",");

  function handleFiles(files) {
    const picked = files?.[0];
    if (picked) onSelect(picked);
  }

  return (
    <div
      className={`dropzone${dragging ? " dragging" : ""}${disabled ? " disabled" : ""}`}
      onClick={() => !disabled && inputRef.current?.click()}
      onDragOver={(e) => {
        e.preventDefault();
        setDragging(true);
      }}
      onDragLeave={() => setDragging(false)}
      onDrop={(e) => {
        e.preventDefault();
        setDragging(false);
        if (!disabled) handleFiles(e.dataTransfer.files);
      }}
      role="button"
      tabIndex={0}
      onKeyDown={(e) => (e.key === "Enter" || e.key === " ") && !disabled && inputRef.current?.click()}
    >
      <input
        ref={inputRef}
        type="file"
        accept={accept}
        hidden
        onChange={(e) => handleFiles(e.target.files)}
      />
      {previewUrl ? (
        <img src={previewUrl} alt="Selected animal" className="preview" />
      ) : (
        <p>Drag and drop an animal photo here, or click to choose a file.</p>
      )}
    </div>
  );
}
