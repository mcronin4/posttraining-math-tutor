"use client";

import { Settings, Trash2, Calculator } from "lucide-react";
import { cn } from "@/lib/utils";

interface HeaderProps {
  onToggleSettings: () => void;
  settingsOpen: boolean;
  onClearChat: () => void;
  messageCount: number;
}

export function Header({
  onToggleSettings,
  settingsOpen,
  onClearChat,
  messageCount,
}: HeaderProps) {
  return (
    <header className="flex items-center justify-between border-b border-surface-200 bg-white/80 px-4 py-3 backdrop-blur-sm">
      {/* Logo and title */}
      <div className="flex items-center gap-3">
        <div className="flex h-10 w-10 items-center justify-center rounded-xl bg-gradient-to-br from-primary-500 to-primary-600 text-white shadow-md">
          <Calculator className="h-5 w-5" />
        </div>
        <div>
          <h1 className="font-display text-xl font-semibold text-surface-900">
            Math Tutor
          </h1>
          <p className="text-xs text-surface-500">Ontario K–12 Curriculum</p>
        </div>
      </div>

      {/* Actions */}
      <div className="flex items-center gap-2">
        {/* Clear chat button */}
        {messageCount > 0 && (
          <button
            onClick={onClearChat}
            className="btn-ghost flex items-center gap-2 text-surface-500 hover:text-surface-700"
            title="Clear chat"
          >
            <Trash2 className="h-4 w-4" />
            <span className="hidden sm:inline text-sm">Clear</span>
          </button>
        )}

        {/* Settings toggle */}
        <button
          onClick={onToggleSettings}
          className={cn(
            "btn-ghost flex items-center gap-2",
            settingsOpen
              ? "text-primary-600 bg-primary-50"
              : "text-surface-500 hover:text-surface-700"
          )}
          title="Toggle settings"
        >
          <Settings className="h-4 w-4" />
          <span className="hidden sm:inline text-sm">Settings</span>
        </button>
      </div>
    </header>
  );
}

