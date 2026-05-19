import { useState, type KeyboardEvent } from "react";
import { CornerDownLeft, Loader2, SendHorizonal, Square } from "lucide-react";

import { Button } from "./Button";

type ChatInputProps = {
  disabled?: boolean;
  streaming?: boolean;
  onSend: (message: string) => void;
  onStop: () => void;
};

export function ChatInput({ disabled, streaming, onSend, onStop }: ChatInputProps) {
  const [value, setValue] = useState("");

  const submit = () => {
    const trimmed = value.trim();
    if (!trimmed || disabled || streaming) {
      return;
    }
    setValue("");
    onSend(trimmed);
  };

  const onKeyDown = (event: KeyboardEvent<HTMLTextAreaElement>) => {
    if (event.key === "Enter" && (event.metaKey || event.ctrlKey)) {
      event.preventDefault();
      submit();
    }
  };

  return (
    <div className="rounded-lg border border-border bg-white p-3 shadow-sm">
      <textarea
        value={value}
        onChange={(event) => setValue(event.target.value)}
        onKeyDown={onKeyDown}
        className="min-h-20 w-full resize-none border-0 bg-transparent text-sm leading-6 outline-none placeholder:text-muted-foreground disabled:opacity-60"
        placeholder={
          disabled
            ? "Upload a dataset before chatting with the agent..."
            : "Ask about structure, anomalies, summaries, charts, exports, or safe transformations..."
        }
        disabled={disabled}
      />
      <div className="mt-3 flex items-center justify-between gap-3">
        <div className="flex items-center gap-1.5 text-xs text-muted-foreground">
          <CornerDownLeft className="h-3.5 w-3.5" />
          <span>Cmd/Ctrl + Enter to send</span>
        </div>
        {streaming ? (
          <Button variant="secondary" onClick={onStop}>
            <Square className="h-4 w-4" />
            Stop
          </Button>
        ) : (
          <Button onClick={submit} disabled={disabled || !value.trim()}>
            {disabled ? <Loader2 className="h-4 w-4 animate-spin" /> : <SendHorizonal className="h-4 w-4" />}
            Send
          </Button>
        )}
      </div>
    </div>
  );
}
