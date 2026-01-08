/**
 * Ontario K-12 Mathematics Curriculum Taxonomy
 *
 * This module provides the curriculum structure for aligning tutoring content
 * with Ontario's mathematics curriculum. The taxonomy includes strands, topics,
 * and keywords for each grade level.
 *
 * Reference: Ontario Mathematics Curriculum (2020)
 */

import type { GradeLevel } from "./schemas";

// =============================================================================
// Types
// =============================================================================

export interface CurriculumTopic {
  id: string;
  name: string;
  keywords: string[];
  description?: string;
}

export interface CurriculumStrand {
  id: string;
  name: string;
  topics: CurriculumTopic[];
}

export interface GradeCurriculum {
  grade: GradeLevel;
  strands: CurriculumStrand[];
}

export interface OntarioMathTaxonomy {
  version: string;
  grades: GradeCurriculum[];
}

// =============================================================================
// Ontario Math Taxonomy (Skeleton)
// =============================================================================

export const ONTARIO_MATH_TAXONOMY: OntarioMathTaxonomy = {
  version: "2020-skeleton",
  grades: [
    // Grade 1
    {
      grade: "1",
      strands: [
        {
          id: "g1-number",
          name: "Number",
          topics: [
            {
              id: "g1-number-counting",
              name: "Counting",
              keywords: [
                "count",
                "counting",
                "numbers to 50",
                "forward",
                "backward",
                "skip counting",
              ],
              description: "Count forward and backward to 50",
            },
            {
              id: "g1-number-quantity",
              name: "Quantity Relationships",
              keywords: [
                "more",
                "less",
                "same",
                "compare",
                "order",
                "greater than",
                "less than",
              ],
              description: "Compare and order quantities",
            },
            {
              id: "g1-number-addition",
              name: "Addition and Subtraction",
              keywords: [
                "add",
                "plus",
                "subtract",
                "minus",
                "sum",
                "difference",
                "take away",
              ],
              description: "Add and subtract single-digit numbers",
            },
          ],
        },
        {
          id: "g1-algebra",
          name: "Algebra",
          topics: [
            {
              id: "g1-algebra-patterns",
              name: "Patterns",
              keywords: ["pattern", "repeat", "extend", "growing", "shrinking"],
              description: "Identify and extend simple patterns",
            },
          ],
        },
        {
          id: "g1-measurement",
          name: "Measurement",
          topics: [
            {
              id: "g1-measurement-length",
              name: "Length",
              keywords: ["long", "short", "tall", "measure", "length", "height"],
              description: "Compare and order objects by length",
            },
          ],
        },
      ],
    },

    // Grade 6
    {
      grade: "6",
      strands: [
        {
          id: "g6-number",
          name: "Number",
          topics: [
            {
              id: "g6-number-fractions",
              name: "Fractions and Decimals",
              keywords: [
                "fraction",
                "decimal",
                "numerator",
                "denominator",
                "equivalent",
                "percent",
                "ratio",
              ],
              description:
                "Operations with fractions and decimals; introduction to ratios",
            },
            {
              id: "g6-number-integers",
              name: "Integers",
              keywords: [
                "integer",
                "negative",
                "positive",
                "opposite",
                "absolute value",
              ],
              description: "Introduction to integers and number line concepts",
            },
            {
              id: "g6-number-operations",
              name: "Operations",
              keywords: [
                "multiply",
                "divide",
                "order of operations",
                "BEDMAS",
                "factor",
                "multiple",
              ],
              description: "Multi-digit multiplication and division",
            },
          ],
        },
        {
          id: "g6-algebra",
          name: "Algebra",
          topics: [
            {
              id: "g6-algebra-expressions",
              name: "Algebraic Expressions",
              keywords: [
                "variable",
                "expression",
                "equation",
                "solve",
                "unknown",
                "substitute",
              ],
              description: "Use variables and evaluate expressions",
            },
          ],
        },
        {
          id: "g6-geometry",
          name: "Geometry",
          topics: [
            {
              id: "g6-geometry-2d",
              name: "2D Geometry",
              keywords: [
                "angle",
                "triangle",
                "quadrilateral",
                "polygon",
                "perpendicular",
                "parallel",
              ],
              description: "Properties of 2D shapes and angle relationships",
            },
            {
              id: "g6-geometry-area",
              name: "Area and Perimeter",
              keywords: [
                "area",
                "perimeter",
                "square units",
                "rectangle",
                "triangle area",
              ],
              description: "Calculate area and perimeter of various shapes",
            },
          ],
        },
      ],
    },

    // Grade 9
    {
      grade: "9",
      strands: [
        {
          id: "g9-number",
          name: "Number",
          topics: [
            {
              id: "g9-number-rational",
              name: "Rational Numbers",
              keywords: [
                "rational",
                "irrational",
                "real number",
                "square root",
                "exponent",
                "power",
              ],
              description:
                "Operations with rational numbers; introduction to irrational numbers",
            },
          ],
        },
        {
          id: "g9-algebra",
          name: "Algebra",
          topics: [
            {
              id: "g9-algebra-linear",
              name: "Linear Relations",
              keywords: [
                "linear",
                "slope",
                "y-intercept",
                "rate of change",
                "graph",
                "equation of line",
              ],
              description: "Represent and analyze linear relations",
            },
            {
              id: "g9-algebra-equations",
              name: "Solving Equations",
              keywords: [
                "solve",
                "equation",
                "isolate",
                "balance",
                "inverse operations",
              ],
              description: "Solve first-degree equations",
            },
            {
              id: "g9-algebra-polynomials",
              name: "Polynomials",
              keywords: [
                "polynomial",
                "term",
                "coefficient",
                "like terms",
                "simplify",
                "expand",
              ],
              description: "Add, subtract, and multiply polynomials",
            },
          ],
        },
        {
          id: "g9-geometry",
          name: "Geometry",
          topics: [
            {
              id: "g9-geometry-analytic",
              name: "Analytic Geometry",
              keywords: [
                "coordinate",
                "distance",
                "midpoint",
                "slope formula",
                "Cartesian plane",
              ],
              description: "Apply analytic geometry concepts",
            },
          ],
        },
      ],
    },

    // Grade 12
    {
      grade: "12",
      strands: [
        {
          id: "g12-calculus",
          name: "Calculus",
          topics: [
            {
              id: "g12-calculus-limits",
              name: "Limits",
              keywords: [
                "limit",
                "continuous",
                "discontinuous",
                "asymptote",
                "infinity",
                "approaches",
              ],
              description: "Evaluate limits and understand continuity",
            },
            {
              id: "g12-calculus-derivatives",
              name: "Derivatives",
              keywords: [
                "derivative",
                "differentiate",
                "rate of change",
                "tangent",
                "chain rule",
                "product rule",
              ],
              description:
                "Apply differentiation rules; interpret derivatives as rates of change",
            },
            {
              id: "g12-calculus-applications",
              name: "Applications of Derivatives",
              keywords: [
                "optimization",
                "maximum",
                "minimum",
                "related rates",
                "curve sketching",
              ],
              description:
                "Solve optimization problems and sketch curves using derivatives",
            },
          ],
        },
        {
          id: "g12-advanced-functions",
          name: "Advanced Functions",
          topics: [
            {
              id: "g12-functions-polynomial",
              name: "Polynomial Functions",
              keywords: [
                "polynomial",
                "degree",
                "roots",
                "zeros",
                "factor theorem",
                "remainder theorem",
              ],
              description: "Analyze polynomial functions and their graphs",
            },
            {
              id: "g12-functions-rational",
              name: "Rational Functions",
              keywords: [
                "rational function",
                "asymptote",
                "hole",
                "domain",
                "range",
              ],
              description: "Analyze rational functions and their behaviour",
            },
            {
              id: "g12-functions-trig",
              name: "Trigonometric Functions",
              keywords: [
                "sine",
                "cosine",
                "tangent",
                "period",
                "amplitude",
                "radian",
                "unit circle",
              ],
              description: "Graph and transform trigonometric functions",
            },
          ],
        },
      ],
    },
  ],
};

// =============================================================================
// Utility Functions
// =============================================================================

/**
 * Get all keywords for a specific grade level
 */
export function getKeywordsForGrade(grade: GradeLevel): string[] {
  const gradeCurriculum = ONTARIO_MATH_TAXONOMY.grades.find(
    (g) => g.grade === grade
  );
  if (!gradeCurriculum) return [];

  const keywords: string[] = [];
  for (const strand of gradeCurriculum.strands) {
    for (const topic of strand.topics) {
      keywords.push(...topic.keywords);
    }
  }
  return [...new Set(keywords)]; // deduplicate
}

/**
 * Find matching topics for a given set of keywords
 */
export function findTopicsByKeywords(
  keywords: string[],
  grade?: GradeLevel
): CurriculumTopic[] {
  const lowerKeywords = keywords.map((k) => k.toLowerCase());
  const matchingTopics: CurriculumTopic[] = [];

  const gradesToSearch = grade
    ? ONTARIO_MATH_TAXONOMY.grades.filter((g) => g.grade === grade)
    : ONTARIO_MATH_TAXONOMY.grades;

  for (const gradeCurriculum of gradesToSearch) {
    for (const strand of gradeCurriculum.strands) {
      for (const topic of strand.topics) {
        const topicKeywords = topic.keywords.map((k) => k.toLowerCase());
        const hasMatch = lowerKeywords.some(
          (kw) =>
            topicKeywords.includes(kw) ||
            topicKeywords.some((tk) => tk.includes(kw) || kw.includes(tk))
        );
        if (hasMatch) {
          matchingTopics.push(topic);
        }
      }
    }
  }

  return matchingTopics;
}

/**
 * Get all strands for a grade level
 */
export function getStrandsForGrade(grade: GradeLevel): CurriculumStrand[] {
  const gradeCurriculum = ONTARIO_MATH_TAXONOMY.grades.find(
    (g) => g.grade === grade
  );
  return gradeCurriculum?.strands ?? [];
}

