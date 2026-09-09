"use client";
import { useEffect, useState } from "react";

type Theme = "light" | "dark";

export default function ThemeToggle() {
    // Starts undefined so the first render matches what the inline script in
    // layout.tsx already put on <html>; reading localStorage during render would
    // mismatch the server-rendered markup and trigger a hydration warning.
    const [theme, setTheme] = useState<Theme | null>(null);

    useEffect(() => {
        setTheme(document.documentElement.classList.contains("dark") ? "dark" : "light");
    }, []);

    const apply = (next: Theme) => {
        document.documentElement.classList.toggle("dark", next === "dark");
        document.documentElement.style.colorScheme = next;
        try {
            localStorage.setItem("cv-theme", next);
        } catch {
            // Private mode / storage disabled: the toggle still works for this
            // session, it just will not be remembered.
        }
        setTheme(next);
    };

    return (
        <button
            type="button"
            onClick={() => apply(theme === "dark" ? "light" : "dark")}
            aria-label={`Switch to ${theme === "dark" ? "light" : "dark"} mode`}
            title={`Switch to ${theme === "dark" ? "light" : "dark"} mode`}
            className="relative flex items-center gap-2 px-3 py-2 rounded-xl border border-line/20 bg-surface/5 hover:bg-surface/10 transition-colors text-muted hover:text-fg"
        >
            {theme === "light" ? (
                <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M20.354 15.354A9 9 0 018.646 3.646 9.003 9.003 0 0012 21a9.003 9.003 0 008.354-5.646z" />
                </svg>
            ) : (
                <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 3v1m0 16v1m9-9h-1M4 12H3m15.364 6.364l-.707-.707M6.343 6.343l-.707-.707m12.728 0l-.707.707M6.343 17.657l-.707.707M16 12a4 4 0 11-8 0 4 4 0 018 0z" />
                </svg>
            )}
            <span className="text-[10px] font-black uppercase tracking-[0.15em] hidden sm:inline">
                {theme === "light" ? "Dark" : "Light"}
            </span>
        </button>
    );
}
