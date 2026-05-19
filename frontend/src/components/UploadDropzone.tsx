import { useRef, useState, type DragEvent } from "react";
import { FileUp, Loader2, UploadCloud } from "lucide-react";

import { cn } from "../lib/utils";
import type { UploadState } from "../hooks/useWorkspace";

type UploadDropzoneProps = {
  disabled?: boolean;
  upload: UploadState;
  onUpload: (files: File[]) => void;
};

export function UploadDropzone({ disabled, upload, onUpload }: UploadDropzoneProps) {
  const inputRef = useRef<HTMLInputElement | null>(null);
  const [dragging, setDragging] = useState(false);

  const handleFiles = (files: FileList | null) => {
    if (!files || disabled) {
      return;
    }
    onUpload(Array.from(files));
  };

  const onDrop = (event: DragEvent<HTMLDivElement>) => {
    event.preventDefault();
    setDragging(false);
    handleFiles(event.dataTransfer.files);
  };

  return (
    <div
      onDragOver={(event) => {
        event.preventDefault();
        setDragging(true);
      }}
      onDragLeave={() => setDragging(false)}
      onDrop={onDrop}
      className={cn(
        "rounded-lg border border-dashed p-4 transition",
        dragging ? "border-teal-500 bg-teal-50" : "border-slate-300 bg-white/72",
        disabled ? "pointer-events-none opacity-60" : "cursor-pointer hover:border-teal-500 hover:bg-white",
      )}
      onClick={() => inputRef.current?.click()}
      role="button"
      tabIndex={0}
      onKeyDown={(event) => {
        if (event.key === "Enter" || event.key === " ") {
          inputRef.current?.click();
        }
      }}
    >
      <input
        ref={inputRef}
        type="file"
        accept=".pkl"
        multiple
        className="hidden"
        onChange={(event) => handleFiles(event.target.files)}
      />
      <div className="flex items-start gap-3">
        <div className="flex h-10 w-10 shrink-0 items-center justify-center rounded-md bg-teal-700 text-white">
          {upload.status === "uploading" ? (
            <Loader2 className="h-5 w-5 animate-spin" />
          ) : (
            <UploadCloud className="h-5 w-5" />
          )}
        </div>
        <div className="min-w-0 flex-1">
          <p className="text-sm font-bold">Upload pickle datasets</p>
          <p className="mt-1 text-xs leading-5 text-muted-foreground">
            Drop one or more trusted `.pkl` files here, or click to browse.
          </p>
          {upload.status !== "idle" ? (
            <div className="mt-3">
              <div className="flex items-center justify-between gap-3 text-xs font-semibold">
                <span className="truncate">{upload.message}</span>
                <span>{upload.status === "uploading" ? `${upload.progress}%` : upload.status}</span>
              </div>
              <div className="mt-2 h-2 overflow-hidden rounded-full bg-slate-200">
                <div
                  className={cn(
                    "h-full rounded-full transition-all",
                    upload.status === "error" ? "bg-rose-500" : "bg-teal-600",
                  )}
                  style={{ width: `${upload.status === "error" ? 100 : upload.progress}%` }}
                />
              </div>
              {upload.filenames.length > 0 ? (
                <div className="mt-2 flex items-center gap-1.5 text-xs text-muted-foreground">
                  <FileUp className="h-3.5 w-3.5" />
                  <span className="truncate">{upload.filenames.join(", ")}</span>
                </div>
              ) : null}
            </div>
          ) : null}
        </div>
      </div>
    </div>
  );
}
