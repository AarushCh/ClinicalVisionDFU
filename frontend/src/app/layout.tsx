import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
    title: "ClinicalVision DFU — Explainable AI Triage",
    description:
        "Explainable diabetic-foot-ulcer triage: CNN classification, Grad-CAM++ attribution and transparent clinical fusion. Research prototype, not a medical device.",
};

// Applied before first paint so a light-mode user never sees a flash of the dark
// theme. It has to be an inline script for that reason -- a React effect runs
// after hydration, which is far too late.
const THEME_SCRIPT = `(function(){try{var t=localStorage.getItem("cv-theme");if(!t){t=window.matchMedia("(prefers-color-scheme: light)").matches?"light":"dark";}document.documentElement.classList.toggle("dark",t!=="light");document.documentElement.style.colorScheme=t==="light"?"light":"dark";}catch(e){document.documentElement.classList.add("dark");}})();`;

export default function RootLayout({
    children,
}: Readonly<{
    children: React.ReactNode;
}>) {
    return (
        <html lang="en" suppressHydrationWarning>
            <head>
                <script dangerouslySetInnerHTML={{ __html: THEME_SCRIPT }} />
            </head>
            <body className="bg-bg text-fg font-sans antialiased">{children}</body>
        </html>
    );
}
