import type { Metadata } from "next";
import { Crimson_Pro, Source_Sans_3 } from "next/font/google";
import "./globals.css";

const sourceSans = Source_Sans_3({
  subsets: ["latin"],
  variable: "--font-geist-sans",
  display: "swap",
});

const crimsonPro = Crimson_Pro({
  subsets: ["latin"],
  variable: "--font-display",
  display: "swap",
});

export const metadata: Metadata = {
  title: "Math Tutor | Learn Math with AI",
  description:
    "An AI-powered math tutoring assistant aligned with Ontario K-12 curriculum. Get hints, check your work, and learn concepts step by step.",
  keywords: ["math tutor", "AI tutor", "Ontario curriculum", "K-12 math", "homework help"],
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en" className={`${sourceSans.variable} ${crimsonPro.variable}`}>
      <body className="min-h-screen bg-gradient-to-br from-surface-50 via-primary-50/30 to-secondary-50/20">
        {children}
      </body>
    </html>
  );
}

