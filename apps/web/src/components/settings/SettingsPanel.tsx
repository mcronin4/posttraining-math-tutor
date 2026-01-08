"use client";

import { Lightbulb, CheckCircle, BookOpen, GraduationCap } from "lucide-react";
import { cn } from "@/lib/utils";
import type {
  TutoringMode,
  GradeLevel,
} from "@/types";
import {
  TUTORING_MODE_LABELS,
  TUTORING_MODE_DESCRIPTIONS,
  GRADE_LEVELS,
} from "@/types";

interface SettingsPanelProps {
  isOpen: boolean;
  mode: TutoringMode;
  onModeChange: (mode: TutoringMode) => void;
  grade: GradeLevel;
  onGradeChange: (grade: GradeLevel) => void;
  dontRevealAnswer: boolean;
  onDontRevealAnswerChange: (value: boolean) => void;
}

const MODE_ICONS: Record<TutoringMode, typeof Lightbulb> = {
  hint: Lightbulb,
  check_step: CheckCircle,
  explain: BookOpen,
};

export function SettingsPanel({
  isOpen,
  mode,
  onModeChange,
  grade,
  onGradeChange,
  dontRevealAnswer,
  onDontRevealAnswerChange,
}: SettingsPanelProps) {
  if (!isOpen) return null;

  return (
    <aside className="w-72 shrink-0 border-r border-surface-200 bg-white/50 backdrop-blur-sm overflow-y-auto">
      <div className="p-4 space-y-6">
        {/* Mode Selection */}
        <section>
          <h2 className="flex items-center gap-2 text-sm font-semibold text-surface-700 mb-3">
            <span className="flex h-5 w-5 items-center justify-center rounded bg-primary-100 text-primary-600">
              <Lightbulb className="h-3 w-3" />
            </span>
            Tutoring Mode
          </h2>
          <div className="space-y-2">
            {(Object.keys(TUTORING_MODE_LABELS) as TutoringMode[]).map((m) => {
              const Icon = MODE_ICONS[m];
              const isSelected = mode === m;
              return (
                <button
                  key={m}
                  onClick={() => onModeChange(m)}
                  className={cn(
                    "w-full rounded-lg border p-3 text-left transition-all duration-200",
                    isSelected
                      ? "border-primary-500 bg-primary-50 shadow-sm"
                      : "border-surface-200 bg-white hover:border-surface-300 hover:bg-surface-50"
                  )}
                >
                  <div className="flex items-center gap-2">
                    <Icon
                      className={cn(
                        "h-4 w-4",
                        isSelected ? "text-primary-600" : "text-surface-400"
                      )}
                    />
                    <span
                      className={cn(
                        "font-medium",
                        isSelected ? "text-primary-700" : "text-surface-700"
                      )}
                    >
                      {TUTORING_MODE_LABELS[m]}
                    </span>
                  </div>
                  <p className="mt-1 text-xs text-surface-500 ml-6">
                    {TUTORING_MODE_DESCRIPTIONS[m]}
                  </p>
                </button>
              );
            })}
          </div>
        </section>

        {/* Grade Selection */}
        <section>
          <h2 className="flex items-center gap-2 text-sm font-semibold text-surface-700 mb-3">
            <span className="flex h-5 w-5 items-center justify-center rounded bg-secondary-100 text-secondary-600">
              <GraduationCap className="h-3 w-3" />
            </span>
            Grade Level
          </h2>
          <select
            value={grade}
            onChange={(e) => onGradeChange(e.target.value as GradeLevel)}
            className="select"
          >
            {GRADE_LEVELS.map((g) => (
              <option key={g} value={g}>
                {g === "K" ? "Kindergarten" : `Grade ${g}`}
              </option>
            ))}
          </select>
          <p className="mt-2 text-xs text-surface-500">
            Responses will be tailored to this grade level.
          </p>
        </section>

        {/* Answer Toggle */}
        <section>
          <h2 className="text-sm font-semibold text-surface-700 mb-3">
            Answer Settings
          </h2>
          <label className="flex items-start gap-3 cursor-pointer group">
            <div className="relative mt-0.5">
              <input
                type="checkbox"
                checked={dontRevealAnswer}
                onChange={(e) => onDontRevealAnswerChange(e.target.checked)}
                className="sr-only peer"
              />
              <div
                className={cn(
                  "toggle",
                  dontRevealAnswer
                    ? "bg-primary-500"
                    : "bg-surface-300"
                )}
              >
                <span
                  className={cn(
                    "toggle-thumb",
                    dontRevealAnswer ? "translate-x-5" : "translate-x-0.5"
                  )}
                />
              </div>
            </div>
            <div>
              <span className="text-sm font-medium text-surface-700 group-hover:text-surface-900">
                Don&apos;t reveal final answer
              </span>
              <p className="text-xs text-surface-500 mt-0.5">
                {dontRevealAnswer
                  ? "The tutor will guide you without giving away the answer"
                  : "The tutor may reveal answers when appropriate"}
              </p>
            </div>
          </label>
        </section>

        {/* Info Card */}
        <div className="rounded-lg bg-gradient-to-br from-primary-50 to-secondary-50 p-4 border border-primary-100">
          <h3 className="text-sm font-semibold text-surface-800 mb-1">
            💡 Tip
          </h3>
          <p className="text-xs text-surface-600 leading-relaxed">
            For the best learning experience, try to solve the problem yourself
            first, then use <strong>Check Step</strong> to verify your work!
          </p>
        </div>
      </div>
    </aside>
  );
}

