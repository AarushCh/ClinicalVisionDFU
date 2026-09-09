import type { Metadata } from "next";
import { Inter, Montserrat } from "next/font/google";
import "./globals.css";

// Inter runs the interface and every number; Helvetica sits behind it in the
// stack as the system fallback. Montserrat is display only — headings and the
// wordmark — and is capped at 600 in globals.css, because its heavy weights
// read childish at this size. Both self-hosted by next/font.
const inter = Inter({
    subsets: ["latin"],
    display: "swap",
    variable: "--font-inter",
});

const montserrat = Montserrat({
    subsets: ["latin"],
    weight: ["400", "500", "600"],
    display: "swap",
    variable: "--font-montserrat",
});

export const metadata: Metadata = {
    title: "ClinicalVision DFU — Explainable AI Triage",
    description:
        "Explainable diabetic-foot-ulcer triage: CNN classification, Grad-CAM++ attribution and transparent clinical fusion. Research prototype, not a medical device.",
};

// Inline so the theme is set before first paint; an effect runs far too late.
const THEME_SCRIPT = `(function(){try{var t=localStorage.getItem("cv-theme");if(!t){t=window.matchMedia("(prefers-color-scheme: light)").matches?"light":"dark";}document.documentElement.classList.toggle("dark",t!=="light");document.documentElement.style.colorScheme=t==="light"?"light":"dark";}catch(e){document.documentElement.classList.add("dark");}})();`;

export default function RootLayout({
    children,
}: Readonly<{
    children: React.ReactNode;
}>) {
    return (
        <html lang="en" className={`${inter.variable} ${montserrat.variable}`} suppressHydrationWarning>
            <head>
                <script dangerouslySetInnerHTML={{ __html: THEME_SCRIPT }} />
            </head>
            <body className="bg-bg text-fg font-sans antialiased">{children}</body>
        </html>
    );
}
