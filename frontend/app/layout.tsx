import type { Metadata } from "next";
import { Inter, JetBrains_Mono } from "next/font/google";
import { Sidebar } from "@/components/Sidebar";
import "./globals.css";

const inter = Inter({
  subsets: ["latin"],
  variable: "--font-inter",
  display: "swap",
});

const jetbrainsMono = JetBrains_Mono({
  subsets: ["latin"],
  variable: "--font-jetbrains",
  display: "swap",
});

const BASE_URL = "https://agentiq.vercel.app";
const DESCRIPTION =
  "AgentIQ — a LangGraph agentic research assistant that routes each question to FAISS retrieval, live Tavily web search, or direct LLM answering, then streams a cited response.";

export const metadata: Metadata = {
  metadataBase: new URL(BASE_URL),
  title: "AgentIQ",
  description: DESCRIPTION,
  icons: { icon: "/favicon.svg" },
  openGraph: {
    title: "AgentIQ",
    description: DESCRIPTION,
    url: BASE_URL,
    siteName: "AgentIQ",
    type: "website",
    locale: "en_US",
  },
  twitter: {
    card: "summary_large_image",
    title: "AgentIQ",
    description: DESCRIPTION,
  },
  robots: { index: true, follow: true },
  alternates: { canonical: BASE_URL },
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en" className={`${inter.variable} ${jetbrainsMono.variable}`}>
      <body>
        <a href="#main" className="skip-nav">
          Skip to main content
        </a>
        <div className="app-shell">
          <Sidebar />
          <main id="main" className="main-content">
            {children}
          </main>
        </div>
      </body>
    </html>
  );
}
