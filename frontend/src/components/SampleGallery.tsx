"use client";
import { useState } from "react";
import { PRESETS, Preset, asset } from "@/lib/config";

const KIND_STYLE: Record<Preset["kind"], { ring: string; chip: string; label: string }> = {
    ulcer: { ring: "border-danger/50", chip: "bg-danger/15 text-danger", label: "Ulcer" },
    healthy: { ring: "border-good/50", chip: "bg-good/15 text-good", label: "Healthy" },
    ood: { ring: "border-warn/50", chip: "bg-warn/15 text-warn", label: "Out of scope" },
};

/**
 * Lets someone try the system with no image and no clinical knowledge of their
 * own. Each card loads a real held-out test image plus a matching patient
 * profile, so the clinical fusion and IWGDF stratification are exercised too.
 */
export default function SampleGallery({
    onPick,
    disabled,
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
            // Fetch the bundled asset and hand it over as a File, so the demo
            // path goes through exactly the same upload flow as a real user's
            // image rather than a special-cased branch.
            const res = await fetch(asset(`/samples/${preset.file}`));
            if (!res.ok) throw new Error(`${res.status}`);
            const blob = await res.blob();
            onPick(preset, new File([blob], preset.file, { type: blob.type || "image/jpeg" }));
        } catch {
            setError("Could not load the sample image. Check that the site's static assets are being served.");
        } finally {
            setLoading(null);
        }
    };

    return (
        <div className="panel rounded-[2rem] p-6">
            <div className="flex items-baseline justify-between gap-3 mb-1">
                <h2 className="text-sm font-black uppercase tracking-[0.15em] text-brand">
                    Try an example
                </h2>
                <span className="text-[10px] text-subtle">no upload needed</span>
            </div>
            <p className="text-xs text-muted mb-5 leading-relaxed">
                Real images from the held-out test split, each with a matching patient
                profile. None was seen during training.
            </p>

            <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-5 gap-3">
                {PRESETS.map((p) => {
                    const s = KIND_STYLE[p.kind];
                    const busy = loading === p.id;
                    return (
                        <button
                            key={p.id}
                            type="button"
                            onClick={() => pick(p)}
                            disabled={disabled || !!loading}
                            title={p.note}
                            className={`group text-left rounded-xl overflow-hidden border-2 ${s.ring} bg-surface2/40 hover:scale-[1.02] transition-transform disabled:opacity-50 disabled:cursor-not-allowed focus:outline-none focus:ring-2 focus:ring-brand/60`}
                        >
                            <div className="relative aspect-square overflow-hidden bg-surface2">
                                <img
                                    src={asset(`/samples/${p.file}`)}
                                    alt={p.label}
                                    loading="lazy"
                                    className="w-full h-full object-cover"
                                />
                                {busy && (
                                    <div className="absolute inset-0 grid place-items-center bg-bg/70">
                                        <svg className="animate-spin h-5 w-5 text-brand" fill="none" viewBox="0 0 24 24">
                                            <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
                                            <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z" />
                                        </svg>
                                    </div>
                                )}
                            </div>
                            <div className="p-2">
                                <span className={`inline-block text-[8px] font-black uppercase tracking-wider px-1.5 py-0.5 rounded ${s.chip}`}>
                                    {s.label}
                                </span>
                                <p className="text-[11px] font-bold mt-1 leading-tight">{p.label}</p>
                                <p className="text-[9px] text-subtle leading-tight">{p.caption}</p>
                            </div>
                        </button>
                    );
                })}
            </div>

            {error && (
                <p role="alert" className="mt-4 text-xs text-danger">{error}</p>
            )}

            <p className="mt-4 text-[10px] text-subtle leading-relaxed border-t border-line/10 pt-3">
                The last card is a whole-foot photograph and is included on purpose: the
                model classifies roughly 100% of out-of-distribution images as ulcers.
                Its result is a demonstration of that limitation, not a finding.
            </p>
        </div>
    );
}
