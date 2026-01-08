"use client";

import { useState, FormEvent, KeyboardEvent } from "react";
import { Send, ChevronDown, ChevronUp } from "lucide-react";
import { cn } from "@/lib/utils";
import type { TutoringMode } from "@/types";
import { TUTORING_MODE_LABELS } from "@/types";

interface ChatInputProps {
  onSend: (question: string, attempt?: string) => void;
  isLoading: boolean;
  mode: TutoringMode;
}

export function ChatInput({ onSend, isLoading, mode }: ChatInputProps) {
  const [question, setQuestion] = useState("");
  const [attempt, setAttempt] = useState("");
  const [showAttempt, setShowAttempt] = useState(false);

  const handleSubmit = (e: FormEvent) => {
    e.preventDefault();
    if (!question.trim() || isLoading) return;

    onSend(question.trim(), attempt.trim() || undefined);
    setQuestion("");
    setAttempt("");
  };

  const handleKeyDown = (e: KeyboardEvent<HTMLTextAreaElement>) => {
    // Submit on Enter (without Shift)
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSubmit(e);
    }
  };

  return (
    <div className="border-t border-surface-200 bg-white/80 backdrop-blur-sm p-4">
      <form onSubmit={handleSubmit} className="mx-auto max-w-3xl">
        {/* Mode indicator */}
        <div className="mb-2 flex items-center gap-2">
          <span className="badge-primary">
            {TUTORING_MODE_LABELS[mode]} Mode
          </span>
          <span className="text-xs text-surface-500">
            {mode === "check_step"
              ? "Share your work for feedback"
              : mode === "hint"
              ? "Get guiding questions"
              : "Get concept explanations"}
          </span>
        </div>

        {/* Main input area */}
        <div className="relative">
          <textarea
            value={question}
            onChange={(e) => setQuestion(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder="Type your math question here..."
            rows={2}
            disabled={isLoading}
            className={cn(
              "textarea pr-12 min-h-[60px]",
              isLoading && "opacity-50 cursor-not-allowed"
            )}
          />
          <button
            type="submit"
            disabled={!question.trim() || isLoading}
            className={cn(
              "absolute right-2 bottom-2 rounded-lg p-2 transition-all duration-200",
              question.trim() && !isLoading
                ? "bg-primary-500 text-white hover:bg-primary-600 shadow-sm"
                : "bg-surface-200 text-surface-400 cursor-not-allowed"
            )}
          >
            <Send className="h-4 w-4" />
          </button>
        </div>

        {/* Attempt section (expandable) */}
        <div className="mt-2">
          <button
            type="button"
            onClick={() => setShowAttempt(!showAttempt)}
            className="flex items-center gap-1 text-sm text-surface-500 hover:text-surface-700 transition-colors"
          >
            {showAttempt ? (
              <ChevronUp className="h-4 w-4" />
            ) : (
              <ChevronDown className="h-4 w-4" />
            )}
            <span>My attempt (optional)</span>
          </button>

          {showAttempt && (
            <div className="mt-2 animate-fade-in">
              <textarea
                value={attempt}
                onChange={(e) => setAttempt(e.target.value)}
                placeholder="Show your work here... What have you tried so far?"
                rows={3}
                disabled={isLoading}
                className={cn(
                  "textarea",
                  isLoading && "opacity-50 cursor-not-allowed"
                )}
              />
              <p className="mt-1 text-xs text-surface-500">
                Sharing your attempt helps the tutor give you better feedback.
              </p>
            </div>
          )}
        </div>

        {/* Help text */}
        <p className="mt-2 text-xs text-surface-400 text-center">
          Press <kbd className="px-1 py-0.5 rounded bg-surface-100 font-mono">Enter</kbd> to send,{" "}
          <kbd className="px-1 py-0.5 rounded bg-surface-100 font-mono">Shift+Enter</kbd> for new line
        </p>
      </form>
    </div>
  );
}

