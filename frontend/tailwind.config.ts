import type { Config } from "tailwindcss";

const token = (name: string) => `rgb(var(--${name}) / <alpha-value>)`;

const config: Config = {
    // Class strategy, not media: the user's explicit choice must win over the OS
    // setting, and it is applied by an inline script in layout.tsx before paint.
    darkMode: "class",
    content: [
        "./src/pages/**/*.{js,ts,jsx,tsx,mdx}",
        "./src/components/**/*.{js,ts,jsx,tsx,mdx}",
        "./src/app/**/*.{js,ts,jsx,tsx,mdx}",
    ],
    theme: {
        extend: {
            // One family everywhere: mono and serif map to it too, so no stray
            // Times or Consolas can appear via font-mono / font-serif.
            fontFamily: {
                sans: ["var(--font-sans)", "ui-sans-serif", "system-ui", "sans-serif"],
                mono: ["var(--font-sans)", "ui-sans-serif", "system-ui", "sans-serif"],
                serif: ["var(--font-sans)", "ui-sans-serif", "system-ui", "sans-serif"],
            },
            colors: {
                bg: token("bg"),
                bgAccent: token("bg-accent"),
                surface: token("surface"),
                surface2: token("surface-2"),
                fg: token("fg"),
                muted: token("fg-muted"),
                subtle: token("fg-subtle"),
                line: token("line"),
                lineStrong: token("line-strong"),
                brand: token("brand"),
                brandFg: token("brand-fg"),
                danger: token("danger"),
                warn: token("warn"),
                good: token("good"),
            },
            keyframes: {
                scan: {
                    "0%": { transform: "translateY(-100%)" },
                    "100%": { transform: "translateY(100%)" },
                },
                shimmer: {
                    "100%": { transform: "translateX(100%)" },
                },
                fadeUp: {
                    "0%": { opacity: "0", transform: "translateY(6px)" },
                    "100%": { opacity: "1", transform: "translateY(0)" },
                },
            },
            animation: {
                fadeUp: "fadeUp 220ms ease-out both",
            },
        },
    },
    plugins: [],
};
export default config;
