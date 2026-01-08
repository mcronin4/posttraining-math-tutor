"use client";

import { useState, useRef, useEffect } from "react";
import { ChatArea } from "@/components/chat/ChatArea";
import { ChatInput } from "@/components/chat/ChatInput";
import { SettingsPanel } from "@/components/settings/SettingsPanel";
import { Header } from "@/components/layout/Header";
import type { ChatMessage, TutoringMode, GradeLevel } from "@/types";

export default function Home() {
  // Chat state
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [isLoading, setIsLoading] = useState(false);

  // Settings state
  const [mode, setMode] = useState<TutoringMode>("hint");
  const [grade, setGrade] = useState<GradeLevel>("6");
  const [dontRevealAnswer, setDontRevealAnswer] = useState(true);
  const [settingsOpen, setSettingsOpen] = useState(true);

  // Refs
  const chatEndRef = useRef<HTMLDivElement>(null);

  // Auto-scroll to bottom when new messages arrive
  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  // Handle sending a message
  const handleSend = async (question: string, attempt?: string) => {
    if (!question.trim()) return;

    // Add user message
    const userMessage: ChatMessage = {
      id: crypto.randomUUID(),
      role: "user",
      content: attempt
        ? `**Question:** ${question}\n\n**My attempt:** ${attempt}`
        : question,
      timestamp: Date.now(),
      metadata: { mode, grade },
    };
    setMessages((prev) => [...prev, userMessage]);
    setIsLoading(true);

    try {
      const response = await fetch("/api/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          question,
          attempt: attempt || undefined,
          mode,
          grade,
          dont_reveal_answer: dontRevealAnswer,
        }),
      });

      if (!response.ok) {
        throw new Error("Failed to get response");
      }

      const data = await response.json();

      // Add assistant message
      const assistantMessage: ChatMessage = {
        id: crypto.randomUUID(),
        role: "assistant",
        content: data.response,
        timestamp: Date.now(),
        metadata: {
          refusal: data.refusal,
        },
      };
      setMessages((prev) => [...prev, assistantMessage]);
    } catch (error) {
      // Add error message
      const errorMessage: ChatMessage = {
        id: crypto.randomUUID(),
        role: "assistant",
        content:
          "Sorry, I had trouble processing your request. Please try again!",
        timestamp: Date.now(),
        metadata: { refusal: false },
      };
      setMessages((prev) => [...prev, errorMessage]);
      console.error("Chat error:", error);
    } finally {
      setIsLoading(false);
    }
  };

  // Handle clearing chat
  const handleClear = () => {
    setMessages([]);
  };

  return (
    <div className="flex h-screen flex-col">
      {/* Header */}
      <Header
        onToggleSettings={() => setSettingsOpen(!settingsOpen)}
        settingsOpen={settingsOpen}
        onClearChat={handleClear}
        messageCount={messages.length}
      />

      {/* Main content */}
      <div className="flex flex-1 overflow-hidden">
        {/* Settings panel (collapsible) */}
        <SettingsPanel
          isOpen={settingsOpen}
          mode={mode}
          onModeChange={setMode}
          grade={grade}
          onGradeChange={setGrade}
          dontRevealAnswer={dontRevealAnswer}
          onDontRevealAnswerChange={setDontRevealAnswer}
        />

        {/* Chat area */}
        <main className="flex flex-1 flex-col overflow-hidden">
          <ChatArea
            messages={messages}
            isLoading={isLoading}
            chatEndRef={chatEndRef}
          />
          <ChatInput onSend={handleSend} isLoading={isLoading} mode={mode} />
        </main>
      </div>
    </div>
  );
}

