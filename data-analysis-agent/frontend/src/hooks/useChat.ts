import { useCallback, useMemo, useRef, useState, type Dispatch, type SetStateAction } from "react";

import { approveConfirmation, rejectConfirmation, streamChat } from "../lib/api";
import type {
  ChatHistoryMessage,
  ChatStreamEvent,
  ExecutionArtifact,
  UpdatedDataset,
} from "../types/api";

export type ChatTraceEvent = {
  id: string;
  type: ChatStreamEvent["type"];
  message?: string;
  code?: string;
  ok?: boolean;
  stdout?: string;
  stderr?: string;
  traceback?: string | null;
  resultSummary?: Record<string, unknown>;
  resultPreview?: unknown;
  updatedDatasets?: UpdatedDataset[];
  severity?: string;
  source?: string;
};

export type ChatMessage = {
  id: string;
  role: "user" | "assistant";
  content: string;
  status?: "streaming" | "done" | "error" | "waiting_confirmation";
  trace: ChatTraceEvent[];
  finalAnswer?: string;
  highlights?: Array<Record<string, unknown>>;
  keyFindings?: string[];
  warnings?: string[];
  stateChanged?: boolean;
  artifacts: ExecutionArtifact[];
};

export type PendingConfirmation = {
  assistantMessageId: string;
  originalMessage: string;
  confirmationId?: string | null;
  message: string;
  code?: string | null;
  mutationSummary?: string | null;
  operationSummary?: string | null;
  title?: string | null;
  datasetName?: string | null;
  expectedEffect?: string | null;
  affectedCount?: number | null;
  currentRowCount?: number | null;
  newRowCount?: number | null;
  stateImpact?: string | null;
  reversible?: boolean | null;
  rollbackNote?: string | null;
  confirmLabel?: string | null;
  cancelLabel?: string | null;
  riskLevel?: string | null;
  affectedDatasetIds?: string[];
};

type UseChatOptions = {
  sessionId?: string | null;
  activeDatasetId?: string | null;
  branchName?: string;
  onStateChanged?: () => void;
};

export function useChat({
  sessionId,
  activeDatasetId,
  branchName = "main",
  onStateChanged,
}: UseChatOptions) {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [isStreaming, setIsStreaming] = useState(false);
  const [pendingConfirmation, setPendingConfirmation] = useState<PendingConfirmation | null>(null);
  const controllerRef = useRef<AbortController | null>(null);

  const artifacts = useMemo(
    () => messages.flatMap((message) => message.artifacts),
    [messages],
  );

  const sendMessage = useCallback(
    async (content: string, options?: { confirmed?: boolean; reuseAssistantId?: string }) => {
      if (!sessionId || !content.trim() || isStreaming) {
        return;
      }

      const trimmed = content.trim();
      const userMessage: ChatMessage = {
        id: crypto.randomUUID(),
        role: "user",
        content: trimmed,
        status: "done",
        trace: [],
        artifacts: [],
      };
      const assistantId = options?.reuseAssistantId ?? crypto.randomUUID();
      const assistantMessage: ChatMessage = {
        id: assistantId,
        role: "assistant",
        content: "",
        status: "streaming",
        trace: [],
        artifacts: [],
      };

      setPendingConfirmation(null);
      setIsStreaming(true);
      setMessages((current) =>
        options?.reuseAssistantId
          ? current.map((message) =>
              message.id === assistantId
                ? { ...assistantMessage, trace: [...message.trace], artifacts: [...message.artifacts] }
                : message,
            )
          : [...current, userMessage, assistantMessage],
      );

      const controller = new AbortController();
      controllerRef.current = controller;

      try {
        await streamChat(
          sessionId,
          {
            message: trimmed,
            active_dataset_id: activeDatasetId,
            branch_name: branchName,
            conversation_history: historyForModel(messages),
            confirmed: options?.confirmed ?? false,
          },
          (event) => {
            handleStreamEvent({
              event,
              assistantId,
              originalMessage: trimmed,
              setMessages,
              setPendingConfirmation,
              onStateChanged,
            });
          },
          controller.signal,
        );
      } catch (error: unknown) {
        const message = error instanceof Error ? error.message : "Chat stream failed";
        setMessages((current) =>
          current.map((item) =>
            item.id === assistantId
              ? {
                  ...item,
                  status: "error",
                  trace: [
                    ...item.trace,
                    {
                      id: crypto.randomUUID(),
                      type: "error",
                      message,
                    },
                  ],
                }
              : item,
          ),
        );
      } finally {
        setIsStreaming(false);
      }
    },
    [activeDatasetId, branchName, isStreaming, messages, onStateChanged, sessionId],
  );

  const confirmPending = useCallback(() => {
    if (!pendingConfirmation) {
      return;
    }

    if (!sessionId || !pendingConfirmation.confirmationId) {
      void sendMessage(pendingConfirmation.originalMessage, {
        confirmed: true,
        reuseAssistantId: pendingConfirmation.assistantMessageId,
      });
      return;
    }

    const assistantId = pendingConfirmation.assistantMessageId;
    setPendingConfirmation(null);
    setIsStreaming(true);
    setMessages((current) =>
      current.map((message) =>
        message.id === assistantId ? { ...message, status: "streaming" } : message,
      ),
    );

    void approveConfirmation(sessionId, pendingConfirmation.confirmationId)
      .then((response) => {
        response.events.forEach((event) => {
          handleStreamEvent({
            event,
            assistantId,
            originalMessage: pendingConfirmation.originalMessage,
            setMessages,
            setPendingConfirmation,
            onStateChanged,
          });
        });
      })
      .catch((error: unknown) => {
        const message = error instanceof Error ? error.message : "Confirmation failed";
        addAssistantError(setMessages, assistantId, message);
      })
      .finally(() => setIsStreaming(false));
  }, [onStateChanged, pendingConfirmation, sendMessage, sessionId]);

  const cancelPending = useCallback(() => {
    if (!pendingConfirmation) {
      return;
    }

    const assistantId = pendingConfirmation.assistantMessageId;
    setPendingConfirmation(null);

    if (!sessionId || !pendingConfirmation.confirmationId) {
      setMessages((current) =>
        current.map((message) =>
          message.id === assistantId
            ? {
                ...message,
                status: "done",
                finalAnswer: "Canceled. I did not run the proposed mutation.",
              }
            : message,
        ),
      );
      return;
    }

    void rejectConfirmation(sessionId, pendingConfirmation.confirmationId)
      .then((response) => {
        response.events.forEach((event) => {
          handleStreamEvent({
            event,
            assistantId,
            originalMessage: pendingConfirmation.originalMessage,
            setMessages,
            setPendingConfirmation,
            onStateChanged,
          });
        });
      })
      .catch((error: unknown) => {
        const message = error instanceof Error ? error.message : "Cancel failed";
        addAssistantError(setMessages, assistantId, message);
      });
  }, [onStateChanged, pendingConfirmation, sessionId]);

  const stop = useCallback(() => {
    controllerRef.current?.abort();
    controllerRef.current = null;
    setIsStreaming(false);
  }, []);

  return {
    messages,
    isStreaming,
    artifacts,
    pendingConfirmation,
    sendMessage,
    confirmPending,
    cancelPending,
    stop,
  };
}

function handleStreamEvent({
  event,
  assistantId,
  originalMessage,
  setMessages,
  setPendingConfirmation,
  onStateChanged,
}: {
  event: ChatStreamEvent;
  assistantId: string;
  originalMessage: string;
  setMessages: Dispatch<SetStateAction<ChatMessage[]>>;
  setPendingConfirmation: Dispatch<SetStateAction<PendingConfirmation | null>>;
  onStateChanged?: () => void;
}) {
  if (event.type === "message_started") {
    return;
  }

  if (event.type === "message_done") {
    setMessages((current) =>
      current.map((message) =>
        message.id === assistantId && message.status === "streaming"
          ? { ...message, status: "done" }
          : message,
      ),
    );
    return;
  }

  if (event.type === "final_answer") {
    if (event.state_changed) {
      onStateChanged?.();
    }
    setMessages((current) =>
      current.map((message) =>
        message.id === assistantId
          ? {
              ...message,
              status: "done",
              finalAnswer: event.answer,
              highlights: event.highlights,
              keyFindings: event.key_findings,
              warnings: event.warnings,
              stateChanged: Boolean(event.state_changed),
            }
          : message,
      ),
    );
    return;
  }

  if (event.type === "artifact_created") {
    setMessages((current) =>
      current.map((message) =>
        message.id === assistantId
          ? { ...message, artifacts: [...message.artifacts, event.artifact] }
          : message,
      ),
    );
    return;
  }

  if (event.type === "confirmation_required") {
    setPendingConfirmation({
      assistantMessageId: assistantId,
      originalMessage,
      confirmationId: event.confirmation_id,
      message: event.message,
      code: event.proposed_code ?? event.code,
      mutationSummary: event.mutation_summary,
      operationSummary: event.operation_summary,
      title: event.title,
      datasetName: event.dataset_name,
      expectedEffect: event.expected_effect,
      affectedCount: event.affected_count,
      currentRowCount: event.current_row_count,
      newRowCount: event.new_row_count,
      stateImpact: event.state_impact,
      reversible: event.reversible,
      rollbackNote: event.rollback_note,
      confirmLabel: event.confirm_label,
      cancelLabel: event.cancel_label,
      riskLevel: event.risk_level,
      affectedDatasetIds: event.affected_dataset_ids,
    });
    setMessages((current) =>
      current.map((message) =>
        message.id === assistantId
          ? {
              ...message,
              status: "waiting_confirmation",
              trace: [
                ...message.trace,
                {
                  id: crypto.randomUUID(),
                  type: event.type,
                  message: event.message,
                  code: event.code ?? undefined,
                },
              ],
            }
          : message,
      ),
    );
    return;
  }

  const traceEvent = toTraceEvent(event);
  if (event.type === "code_result_summary" && event.updated_datasets?.length) {
    onStateChanged?.();
  }
  setMessages((current) =>
    current.map((message) =>
      message.id === assistantId ? { ...message, trace: [...message.trace, traceEvent] } : message,
    ),
  );
}

function toTraceEvent(event: ChatStreamEvent): ChatTraceEvent {
  if (event.type === "trace") {
    return { id: crypto.randomUUID(), type: event.type, message: event.message };
  }

  if (event.type === "code_started") {
    return { id: crypto.randomUUID(), type: event.type, code: event.code };
  }

  if (event.type === "code_result_summary") {
    return {
      id: crypto.randomUUID(),
      type: event.type,
      ok: event.ok,
      stdout: event.stdout,
      stderr: event.stderr,
      traceback: event.traceback,
      resultSummary: event.result_summary,
      resultPreview: event.result_preview,
      updatedDatasets: event.updated_datasets,
    };
  }

  if (event.type === "error") {
    return { id: crypto.randomUUID(), type: event.type, message: event.message };
  }

  if (event.type === "verifier_result") {
    return {
      id: crypto.randomUUID(),
      type: event.type,
      message: event.message,
      severity: event.severity,
      source: event.source,
    };
  }

  return { id: crypto.randomUUID(), type: event.type };
}

function addAssistantError(
  setMessages: Dispatch<SetStateAction<ChatMessage[]>>,
  assistantId: string,
  message: string,
) {
  setMessages((current) =>
    current.map((item) =>
      item.id === assistantId
        ? {
            ...item,
            status: "error",
            trace: [
              ...item.trace,
              {
                id: crypto.randomUUID(),
                type: "error",
                message,
              },
            ],
          }
        : item,
    ),
  );
}

function historyForModel(messages: ChatMessage[]): ChatHistoryMessage[] {
  return messages
    .filter((message) => message.role === "user" || message.finalAnswer)
    .slice(-8)
    .map((message) => ({
      role: message.role,
      content: message.role === "assistant" ? message.finalAnswer ?? "" : message.content,
    }));
}
