import { useCallback, useEffect, useMemo, useRef, useState, type Dispatch, type SetStateAction } from "react";

import { approveConfirmation, listMessages, rejectConfirmation, streamChat } from "../lib/api";
import type {
  ChatHistoryMessage,
  ChatStreamEvent,
  ExecutionArtifact,
  PersistedChatMessage,
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
  status?: "streaming" | "done" | "error" | "waiting_confirmation" | "waiting_clarification";
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

export type PendingClarification = {
  assistantMessageId: string;
  originalMessage: string;
  title?: string | null;
  message: string;
  options: Array<{
    id?: string;
    label: string;
    description?: string | null;
    message?: string | null;
  }>;
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
  const [pendingClarification, setPendingClarification] = useState<PendingClarification | null>(null);
  const controllerRef = useRef<AbortController | null>(null);
  const loadedSessionRef = useRef<string | null>(null);

  useEffect(() => {
    if (!sessionId || isStreaming) {
      return;
    }

    let mounted = true;
    loadedSessionRef.current = sessionId;
    setPendingConfirmation(null);
    setPendingClarification(null);
    listMessages(sessionId)
      .then((response) => {
        if (!mounted || loadedSessionRef.current !== sessionId) {
          return;
        }
        setMessages(response.messages.map(persistedMessageToChatMessage));
        const restored = restorePendingAction(response.messages);
        setPendingConfirmation(restored.confirmation);
        setPendingClarification(restored.clarification);
      })
      .catch(() => {
        if (mounted && loadedSessionRef.current === sessionId) {
          setMessages([]);
        }
      });

    return () => {
      mounted = false;
    };
  }, [isStreaming, sessionId]);

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
      setPendingClarification(null);
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
              setPendingClarification,
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
            setPendingClarification,
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

  const chooseClarification = useCallback((option: PendingClarification["options"][number]) => {
    if (!pendingClarification) {
      return;
    }
    const assistantId = pendingClarification.assistantMessageId;
    const nextMessage = option.message || option.label;
    setPendingClarification(null);
    setMessages((current) =>
      current.map((message) =>
        message.id === assistantId
          ? {
              ...message,
              status: "done",
              finalAnswer: `Clarification selected: ${option.label}`,
              stateChanged: false,
            }
          : message,
      ),
    );
    void sendMessage(nextMessage);
  }, [pendingClarification, sendMessage]);

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
            setPendingClarification,
            onStateChanged,
          });
        });
      })
      .catch((error: unknown) => {
        const message = error instanceof Error ? error.message : "Cancel failed";
        addAssistantError(setMessages, assistantId, message);
      });
  }, [onStateChanged, pendingConfirmation, sessionId]);

  const cancelClarification = useCallback(() => {
    if (!pendingClarification) {
      return;
    }
    const assistantId = pendingClarification.assistantMessageId;
    setPendingClarification(null);
    setMessages((current) =>
      current.map((message) =>
        message.id === assistantId
          ? {
              ...message,
              status: "done",
              finalAnswer: "Canceled. I did not apply a cleaning rule, and the dataset state was left unchanged.",
              stateChanged: false,
            }
          : message,
      ),
    );
  }, [pendingClarification]);

  const stop = useCallback(() => {
    controllerRef.current?.abort();
    controllerRef.current = null;
    setIsStreaming(false);
  }, []);

  const clearMessages = useCallback(() => {
    setMessages([]);
    setPendingConfirmation(null);
    setPendingClarification(null);
  }, []);

  return {
    messages,
    isStreaming,
    artifacts,
    pendingConfirmation,
    pendingClarification,
    sendMessage,
    confirmPending,
    cancelPending,
    chooseClarification,
    cancelClarification,
    clearMessages,
    stop,
  };
}

function persistedMessageToChatMessage(message: PersistedChatMessage): ChatMessage {
  return {
    id: message.id,
    role: message.role === "user" ? "user" : "assistant",
    content: message.content,
    status: normalizeStatus(message.status),
    trace: (message.trace_events ?? []).map((event) => ({
      id: event.id,
      type: event.type as ChatStreamEvent["type"],
      message: event.message ?? undefined,
      code: event.code ?? undefined,
      ok: event.ok ?? undefined,
      stdout: event.stdout ?? undefined,
      stderr: event.stderr ?? undefined,
      traceback: event.traceback ?? undefined,
      resultSummary: event.result_summary ?? undefined,
      resultPreview: event.result_preview,
      updatedDatasets: event.updated_datasets,
      severity: event.severity ?? undefined,
      source: event.source ?? undefined,
    })),
    finalAnswer: message.final_answer ?? undefined,
    highlights: message.highlights ?? [],
    keyFindings: message.key_findings ?? [],
    warnings: message.warnings ?? [],
    stateChanged: message.state_changed ?? undefined,
    artifacts: message.artifacts ?? [],
  };
}

function restorePendingAction(messages: PersistedChatMessage[]): {
  confirmation: PendingConfirmation | null;
  clarification: PendingClarification | null;
} {
  for (let index = messages.length - 1; index >= 0; index -= 1) {
    const message = messages[index];
    if (message.role !== "assistant" || !message.pending_action) {
      continue;
    }
    const previousUser = [...messages.slice(0, index)].reverse().find((item) => item.role === "user");
    const originalMessage = previousUser?.content ?? "";
    const action = message.pending_action;
    if (action.type === "confirmation_required" && message.status === "waiting_confirmation") {
      return {
        confirmation: {
          assistantMessageId: message.id,
          originalMessage,
          confirmationId: stringOrNull(action.confirmation_id),
          message: String(action.message ?? "Confirmation required"),
          code: stringOrNull(action.proposed_code) ?? stringOrNull(action.code),
          mutationSummary: stringOrNull(action.mutation_summary),
          operationSummary: stringOrNull(action.operation_summary),
          title: stringOrNull(action.title),
          datasetName: stringOrNull(action.dataset_name),
          expectedEffect: stringOrNull(action.expected_effect),
          affectedCount: numberOrNull(action.affected_count),
          currentRowCount: numberOrNull(action.current_row_count),
          newRowCount: numberOrNull(action.new_row_count),
          stateImpact: stringOrNull(action.state_impact),
          reversible: booleanOrNull(action.reversible),
          rollbackNote: stringOrNull(action.rollback_note),
          confirmLabel: stringOrNull(action.confirm_label),
          cancelLabel: stringOrNull(action.cancel_label),
          riskLevel: stringOrNull(action.risk_level),
          affectedDatasetIds: Array.isArray(action.affected_dataset_ids)
            ? action.affected_dataset_ids.filter((item): item is string => typeof item === "string")
            : undefined,
        },
        clarification: null,
      };
    }
    if (action.type === "clarification_required" && message.status === "waiting_clarification") {
      const options = Array.isArray(action.options)
        ? action.options
            .filter((item): item is Record<string, unknown> => item !== null && typeof item === "object")
            .map((item) => ({
              id: stringOrNull(item.id) ?? undefined,
              label: String(item.label ?? "Option"),
              description: stringOrNull(item.description),
              message: stringOrNull(item.message),
            }))
        : [];
      return {
        confirmation: null,
        clarification: {
          assistantMessageId: message.id,
          originalMessage,
          title: stringOrNull(action.title),
          message: String(action.message ?? message.final_answer ?? "Clarification needed"),
          options,
        },
      };
    }
  }
  return { confirmation: null, clarification: null };
}

function stringOrNull(value: unknown): string | null {
  return typeof value === "string" ? value : null;
}

function numberOrNull(value: unknown): number | null {
  return typeof value === "number" ? value : null;
}

function booleanOrNull(value: unknown): boolean | null {
  return typeof value === "boolean" ? value : null;
}

function normalizeStatus(status: string): ChatMessage["status"] {
  if (status === "streaming" || status === "error" || status === "waiting_confirmation" || status === "waiting_clarification") {
    return status;
  }
  return "done";
}

function handleStreamEvent({
  event,
  assistantId,
  originalMessage,
  setMessages,
  setPendingConfirmation,
  setPendingClarification,
  onStateChanged,
}: {
  event: ChatStreamEvent;
  assistantId: string;
  originalMessage: string;
  setMessages: Dispatch<SetStateAction<ChatMessage[]>>;
  setPendingConfirmation: Dispatch<SetStateAction<PendingConfirmation | null>>;
  setPendingClarification: Dispatch<SetStateAction<PendingClarification | null>>;
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
    setPendingClarification(null);
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
            }
          : message,
      ),
    );
    return;
  }

  if (event.type === "clarification_required") {
    setPendingConfirmation(null);
    setPendingClarification({
      assistantMessageId: assistantId,
      originalMessage,
      title: event.title,
      message: event.message,
      options: event.options ?? [],
    });
    setMessages((current) =>
      current.map((message) =>
        message.id === assistantId
          ? {
              ...message,
              status: "waiting_clarification",
              finalAnswer: event.message,
              stateChanged: false,
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
