/**
 * @math-tutor/core
 *
 * Shared types, schemas, and utilities for the LLM Math Tutor project.
 */

// Re-export all schemas and types
export * from "./schemas";
export * from "./curriculum";

// Export prompt template paths (to be loaded at runtime)
export const PROMPT_TEMPLATES = {
  tutorSystemPrompt: "prompts/tutor_system_prompt.txt",
  refusalPolicy: "prompts/refusal_policy.md",
} as const;

