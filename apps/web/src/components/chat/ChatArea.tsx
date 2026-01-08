"use client";

import { RefObject } from "react";
import { User, Bot, Sparkles } from "lucide-react";
import { cn, formatTime } from "@/lib/utils";
import type { ChatMessage } from "@/types";

interface ChatAreaProps {
  messages: ChatMessage[];
  isLoading: boolean;
  chatEndRef: RefObject<HTMLDivElement>;
}

export function ChatArea({ messages, isLoading, chatEndRef }: ChatAreaProps) {
  if (messages.length === 0 && !isLoading) {
    return <EmptyState />;
  }

  return (
    <div className="flex-1 overflow-y-auto px-4 py-6">
      <div className="mx-auto max-w-3xl space-y-4">
        {messages.map((message) => (
          <MessageBubble key={message.id} message={message} />
        ))}
        {isLoading && <TypingIndicator />}
        <div ref={chatEndRef} />
      </div>
    </div>
  );
}

function EmptyState() {
  return (
    <div className="flex-1 flex items-center justify-center p-8">
      <div className="text-center max-w-md animate-fade-in">
        <div className="mx-auto mb-6 flex h-20 w-20 items-center justify-center rounded-2xl bg-gradient-to-br from-primary-100 to-secondary-100 shadow-inner">
          <Sparkles className="h-10 w-10 text-primary-500" />
        </div>
        <h2 className="font-display text-2xl font-semibold text-surface-800 mb-2">
          Ready to Learn Math!
        </h2>
        <p className="text-surface-600 mb-6 leading-relaxed">
          Ask a math question, and I&apos;ll help guide you to the answer. You
          can also share your attempt and I&apos;ll check your work.
        </p>
        <div className="grid grid-cols-1 sm:grid-cols-3 gap-3 text-sm">
          <ExamplePrompt text="What is 24 × 15?" />
          <ExamplePrompt text="How do I solve x + 5 = 12?" />
          <ExamplePrompt text="Explain fractions" />
        </div>
      </div>
    </div>
  );
}

function ExamplePrompt({ text }: { text: string }) {
  return (
    <div className="rounded-lg border border-surface-200 bg-white/80 px-3 py-2 text-surface-600 hover:border-primary-300 hover:bg-primary-50 transition-colors cursor-default">
      &ldquo;{text}&rdquo;
    </div>
  );
}

function MessageBubble({ message }: { message: ChatMessage }) {
  const isUser = message.role === "user";

  return (
    <div
      className={cn(
        "flex gap-3 animate-slide-up",
        isUser ? "flex-row-reverse" : "flex-row"
      )}
    >
      {/* Avatar */}
      <div
        className={cn(
          "flex h-8 w-8 shrink-0 items-center justify-center rounded-full",
          isUser
            ? "bg-primary-500 text-white"
            : "bg-surface-100 text-surface-600"
        )}
      >
        {isUser ? <User className="h-4 w-4" /> : <Bot className="h-4 w-4" />}
      </div>

      {/* Message content */}
      <div
        className={cn(
          "flex flex-col gap-1",
          isUser ? "items-end" : "items-start"
        )}
      >
        <div
          className={cn(
            isUser ? "message-user" : "message-assistant",
            message.metadata?.refusal && "border-l-4 border-amber-400"
          )}
        >
          <MessageContent content={message.content} />
        </div>
        <span className="text-xs text-surface-400 px-1">
          {formatTime(message.timestamp)}
        </span>
      </div>
    </div>
  );
}

function MessageContent({ content }: { content: string }) {
  // Simple markdown-like rendering for bold text
  const parts = content.split(/(\*\*[^*]+\*\*)/g);

  return (
    <p className="whitespace-pre-wrap">
      {parts.map((part, i) => {
        if (part.startsWith("**") && part.endsWith("**")) {
          return (
            <strong key={i} className="font-semibold">
              {part.slice(2, -2)}
            </strong>
          );
        }
        return part;
      })}
    </p>
  );
}

function TypingIndicator() {
  return (
    <div className="flex gap-3 animate-fade-in">
      <div className="flex h-8 w-8 shrink-0 items-center justify-center rounded-full bg-surface-100 text-surface-600">
        <Bot className="h-4 w-4" />
      </div>
      <div className="message-assistant">
        <div className="typing-indicator">
          <span className="typing-dot" style={{ animationDelay: "0ms" }} />
          <span className="typing-dot" style={{ animationDelay: "150ms" }} />
          <span className="typing-dot" style={{ animationDelay: "300ms" }} />
        </div>
      </div>
    </div>
  );
}

