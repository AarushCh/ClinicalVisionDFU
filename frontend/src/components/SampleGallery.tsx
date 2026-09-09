"use client";
import { useState } from "react";
import { PRESETS, Preset, asset } from "@/lib/config";

const KIND: Record<Preset["kind"], { ring: string; dot: string }> = {
    ulcer: { ring: "hover:border-danger/60", dot: "bg-danger" },
    healthy: { ring: "hover:border-good/60", dot: "bg-good" },
    ood: { ring: "hover:border-warn/60", dot: "bg-warn" },
};

/**
 * Try the system with no image of your own: each card loads a held-out test
 * image plus a matching patient profile, so the fusion is exercised too.
 */
export default function SampleGallery({
    onPick, disabled,
}: {
    onPick: (preset: Preset, file: File) => void;
    disabled?: boolean;
}) {
    const [loading, setLoading] = useState<string | null>(null);
    const [error, setError] = useState<string | null>(null);

    const pick = async (preset: Preset) => {
        if (disabled || loading) return;
        setLoading(preset.id);
        setError(null);
        try {
            // Handed over as a File, so the demo takes the same path as an upload.
            const res = await fetch(asset(`/samples/${preset.file}`));
            if (!res.ok) throw new Error(`${res.status}`);
            const blob = await res.blob();
            onPick(preset, new File([blob], preset.file, { type: blob.type || "image/jpeg" }));
        } catch {
            setError("Could not load the sample. Check that static assets are being served.");
        } finally {
            setLoading(null);
        }
    };

    return (
        <div className="panel rounded-2xl p-5">
            <div className="flex items-center justify-between gap-3 mb-3">
                <h2 className="text-[11px] font-black uppercase tracking-[0.18em] flex items-center gap-2">
                    <span className="sec-num">02</span>
                    <span className="text-muted">Examples</span>
                </h2>
                <span className="text-[10px] text-subtle">no upload needed</span>
            </div>

            <div className="grid grid-cols-3 gap-2">
                {PRESETS.map((p) => {
                    const k = KIND[p.kind];
                    const busy = loading === p.id;
                    return (
                        <button
                            key={p.id}
                            type="button"
                            onClick={() => pick(p)}
                            disabled={disabled || !!loading}
                            title={`${p.label} — ${p.note}`}
                            className={`group text-left rounded-lg overflow-hidden border border-line/20 ${k.ring} bg-surface2/40 transition-colors disabled:opacity-50 disabled:cursor-not-allowed focus:outline-none focus:ring-2 focus:ring-brand/60`}
                        >
                            <div className="relative aspect-square overflow-hidden bg-surface2">
                                <img
                                    src={asset(`/samples/${p.file}`)}
                                    alt={p.label}
                                    loading="lazy"
                                    className="w-full h-full object-cover group-hover:scale-105 transition-transform duration-300"
                                />
                                {busy && (
                                    <div className="absolute inset-0 grid place-items-center bg-bg/70">
                                        <svg className="animate-spin h-4 w-4 text-brand" fill="none" viewBox="0 0 24 24">
                                            <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
                                            <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z" />
                                        </svg>
                                    </div>
                                )}
                            </div>
                            <p className="px-1.5 py-1.5 text-[10px] font-bold leading-tight flex items-center gap-1.5">
                                <span className={`w-1.5 h-1.5 rounded-full shrink-0 ${k.dot}`} />
                                <span className="truncate">{p.label}</span>
                            </p>
                        </button>
                    );
                })}
            </div>

            {error && <p role="alert" className="mt-3 text-[11px] text-danger">{error}</p>}

            <p className="mt-3 text-[10px] text-subtle leading-snug">
                Held-out test images, none seen in training. The whole-foot card is
                deliberately out of scope — it demonstrates the failure, it is not a finding.
            </p>
        </div>
    );
}
