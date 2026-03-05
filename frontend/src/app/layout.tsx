import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
    title: "ClinicalVision DFU",
    description: "Explainable AI Clinical Support System",
};

export default function RootLayout({
    children,
}: Readonly<{
    children: React.ReactNode;
}>) {
    return (
        <html lang="en">
            <body className="bg-slate-50 text-slate-900 font-sans">{children}</body>
        </html>
    );
}