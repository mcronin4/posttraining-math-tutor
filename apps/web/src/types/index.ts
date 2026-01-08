/**
 * Shared types for the Math Tutor frontend.
 *
 * TODO: Import these from @math-tutor/core once the package is built.
 * For now, we duplicate the essential types to avoid build dependency issues.
 */

// =============================================================================
// Tutoring Mode
// =============================================================================

export type TutoringMode = "hint" | "check_step" | "explain";

export const TUTORING_MODE_LABELS: Record<TutoringMode, string> = {
  hint: "Hint",
  check_step: "Check Step",
  explain: "Explain",
};

export const TUTORING_MODE_DESCRIPTIONS: Record<TutoringMode, string> = {
  hint: "Get guiding questions to discover the answer yourself",
  check_step: "Have your work checked and get feedback",
  explain: "Get the concept explained without the direct answer",
};

// =============================================================================
// Grade Level
// =============================================================================

export type GradeLevel =
  | "K"
  | "1"
  | "2"
  | "3"
  | "4"
  | "5"
  | "6"
  | "7"
  | "8"
  | "9"
  | "10"
  | "11"
  | "12";

export const GRADE_LEVELS: GradeLevel[] = [
  "K",
  "1",
  "2",
  "3",
  "4",
  "5",
  "6",
  "7",
  "8",
  "9",
  "10",
  "11",
  "12",
];

// =============================================================================
// Chat Types
// =============================================================================

export type MessageRole = "user" | "assistant" | "system";

export interface ChatMessage {
  id: string;
  role: MessageRole;
  content: string;
  timestamp: number;
  metadata?: {
    mode?: TutoringMode;
    grade?: GradeLevel;
    refusal?: boolean;
  };
}

// =============================================================================
// API Types
// =============================================================================

export interface ChatRequest {
  question: string;
  attempt?: string;
  mode: TutoringMode;
  grade: GradeLevel;
  dont_reveal_answer: boolean;
  topic_tags?: string[];
}

export interface ChatResponse {
  response: string;
  refusal: boolean;
  citations?: string[];
  debug?: {
    selected_policy: string;
  };
}

