"use client";
import { ReactNode } from "react";

/** Shared bento primitives: rings, mini bars, tiles and section headers. */

export const TONE = {
    danger: { stroke: "rgb(var(--danger))", text: "text-danger", glow: "glow-danger", chip: "bg-danger/15 text-danger border-danger/30" },
    warn: { stroke: "rgb(var(--warn))", text: "text-warn", glow: "glow-warn", chip: "bg-warn/15 text-warn border-warn/30" },
    good: { stroke: "rgb(var(--good))", text: "text-good", glow: "glow-good", chip: "bg-good/15 text-good border-good/30" },
    brand: { stroke: "rgb(var(--brand))", text: "text-brand", glow: "", chip: "bg-brand/15 text-brand border-brand/30" },
    muted: { stroke: "rgb(var(--fg-subtle))", text: "text-subtle", glow: "", chip: "bg-surface2 text-subtle border-line/20" },
} as const;

export type Tone = keyof typeof TONE;

export const toneForRisk = (risk?: string): Tone =>
    risk === "HIGH" ? "danger" : risk === "MEDIUM" ? "warn" : risk === "LOW" ? "good" : "muted";

/** Donut gauge. `value` is 0..1. */
export function Ring({
    value, label, caption, tone = "brand", size = 104, stroke = 9,
}: {
    value: number; label: string; caption?: string; tone?: Tone; size?: number; stroke?: number;
}) {
    const v = Math.max(0, Math.min(1, Number.isFinite(value) ? value : 0));
    const r = (size - stroke) / 2;
    const circ = 2 * Math.PI * r;
    const t = TONE[tone];

    return (
        <div className="flex flex-col items-center gap-1.5">
            <div className="relative" style={{ width: size, height: size }}>
                <svg width={size} height={size} role="img" aria-label={`${label}: ${(v * 100).toFixed(1)}%`}>
                    <circle
                        className="ring-track" cx={size / 2} cy={size / 2} r={r}
                        fill="none" stroke="rgb(var(--fg-subtle) / 0.18)" strokeWidth={stroke}
                    />
                    <circle
                        className="ring-value" cx={size / 2} cy={size / 2} r={r}
                        fill="none" stroke={t.stroke} strokeWidth={stroke} strokeLinecap="round"
                        strokeDasharray={circ} strokeDashoffset={circ * (1 - v)}
                    />
                </svg>
                <div className="absolute inset-0 grid place-items-center">
                    <span className={`text-lg font-black tabular-nums ${t.text}`}>
                        {(v * 100).toFixed(0)}<span className="text-[10px] align-super">%</span>
                    </span>
                </div>
            </div>
            <p className="text-[9px] font-black uppercase tracking-[0.15em] text-subtle text-center leading-tight">
                {label}
            </p>
            {caption && <p className="text-[9px] text-subtle text-center leading-tight">{caption}</p>}
        </div>
    );
}

/** Signed horizontal bar for a Shapley contribution. */
export function SignedBar({ pct, positive }: { pct: number; positive: boolean }) {
    return (
        <div className="h-1.5 w-full bg-surface2 rounded-full overflow-hidden">
            <div
                className={`h-full rounded-full ${positive
                    ? "bg-gradient-to-r from-danger to-orange-500"
                    : "bg-gradient-to-r from-good to-teal-400"}`}
                style={{ width: `${Math.max(Math.min(pct, 100), 1.5)}%` }}
            />
        </div>
    );
}

/** Tiny bar chart, used for the history risk trend. */
export function MiniBars({ values, tones }: { values: number[]; tones?: Tone[] }) {
    if (!values.length) return null;
    return (
        <div className="flex items-end gap-[3px] h-10" aria-hidden>
            {values.map((v, i) => (
                <div
                    key={i}
                    className="flex-1 min-w-[3px] rounded-sm"
                    style={{
                        height: `${Math.max(6, Math.min(100, v * 100))}%`,
                        background: TONE[tones?.[i] ?? "brand"].stroke,
                        opacity: 0.55 + 0.45 * (i / Math.max(1, values.length - 1)),
                    }}
                />
            ))}
        </div>
    );
}

export function SectionHead({ n, title, right }: { n: string; title: string; right?: ReactNode }) {
    return (
        <div className="flex items-center justify-between gap-3 mb-3">
            <h3 className="text-[11px] font-black uppercase tracking-[0.18em] flex items-center gap-2">
                <span className="sec-num">{n}</span>
                <span className="text-muted">{title}</span>
            </h3>
            {right}
        </div>
    );
}

export function Tile({
    children, className = "", neon = false, grid = false,
}: {
    children: ReactNode; className?: string; neon?: boolean; grid?: boolean;
}) {
    return (
        <div className={`panel-inset rounded-2xl p-4 ${neon ? "neon neon-soft" : ""} ${grid ? "bento-grid-bg" : ""} ${className}`}>
            {children}
        </div>
    );
}

export function Stat({ label, value, hint, tone = "muted" }: {
    label: string; value: ReactNode; hint?: string; tone?: Tone;
}) {
    return (
        <Tile>
            <p className="text-[9px] font-black text-subtle uppercase tracking-[0.15em] leading-tight">{label}</p>
            <p className={`text-xl font-black mt-1 leading-none tabular-nums ${TONE[tone].text}`}>{value}</p>
            {hint && <p className="text-[10px] text-subtle leading-tight mt-1.5">{hint}</p>}
        </Tile>
    );
}

export function Chip({ children, tone = "muted" }: { children: ReactNode; tone?: Tone }) {
    return (
        <span className={`inline-flex items-center gap-1.5 px-2.5 py-1 rounded-full border text-[10px] font-black uppercase tracking-[0.12em] ${TONE[tone].chip}`}>
            {children}
        </span>
    );
}
