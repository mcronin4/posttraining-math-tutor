/**
 * @math-tutor/core - Shared schemas and types
 *
 * These schemas define the contract between frontend, backend, and evaluation.
 * Use Zod for runtime validation and TypeScript inference.
 */

import { z } from "zod";

// =============================================================================
// Tutoring Mode
// =============================================================================

export const TutoringMode = z.enum(["hint", "check_step", "explain"]);
export type TutoringMode = z.infer<typeof TutoringMode>;

export const TUTORING_MODE_DESCRIPTIONS: Record<TutoringMode, string> = {
  hint: "Provide guiding questions to help discover the answer",
  check_step: "Validate student's work and suggest next steps",
  explain: "Explain the concept without revealing the answer",
};

// =============================================================================
// Grade Level
// =============================================================================

export const GradeLevel = z.enum([
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
]);
export type GradeLevel = z.infer<typeof GradeLevel>;

// =============================================================================
// Chat Request/Response
// =============================================================================

export const ChatRequest = z.object({
  question: z.string().min(1, "Question is required"),
  attempt: z.string().optional(),
  mode: TutoringMode,
  grade: GradeLevel,
  dont_reveal_answer: z.boolean().default(true),
  topic_tags: z.array(z.string()).optional(),
});
export type ChatRequest = z.infer<typeof ChatRequest>;

export const ChatResponse = z.object({
  response: z.string(),
  refusal: z.boolean(),
  citations: z.array(z.string()).optional(),
  debug: z
    .object({
      selected_policy: z.string(),
    })
    .optional(),
});
export type ChatResponse = z.infer<typeof ChatResponse>;

// =============================================================================
// Chat Message (for UI state)
// =============================================================================

export const MessageRole = z.enum(["user", "assistant", "system"]);
export type MessageRole = z.infer<typeof MessageRole>;

export const ChatMessage = z.object({
  id: z.string(),
  role: MessageRole,
  content: z.string(),
  timestamp: z.number(),
  metadata: z
    .object({
      mode: TutoringMode.optional(),
      grade: GradeLevel.optional(),
      refusal: z.boolean().optional(),
    })
    .optional(),
});
export type ChatMessage = z.infer<typeof ChatMessage>;

// =============================================================================
// Evaluation Rubric
// =============================================================================

export const TutorRubricResult = z.object({
  no_answer_reveal: z.boolean(),
  socratic_step: z.boolean(),
  correct_direction: z.boolean().nullable(), // null = not evaluated
  on_topic: z.boolean(),
});
export type TutorRubricResult = z.infer<typeof TutorRubricResult>;

export const EvalResult = z.object({
  id: z.string(),
  input: z.string(),
  expected: z.string().optional(),
  actual: z.string(),
  metrics: z.record(z.union([z.number(), z.boolean(), z.null()])),
});
export type EvalResult = z.infer<typeof EvalResult>;

// =============================================================================
// Training Data Schema
// =============================================================================

export const TrainingExample = z.object({
  id: z.string(),
  grade: GradeLevel,
  topic_tags: z.array(z.string()),
  prompt: z.string(),
  response: z.string(),
  metadata: z.object({
    source: z.string(),
    difficulty: z.enum(["easy", "medium", "hard"]).optional(),
    mode: TutoringMode.optional(),
  }),
});
export type TrainingExample = z.infer<typeof TrainingExample>;

