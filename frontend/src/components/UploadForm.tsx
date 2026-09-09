"use client";
import { useState, useCallback, useEffect } from "react";
import axios from "axios";
import { API_URL, Preset } from "@/lib/config";

type Numeric = { age: string; bmi: string; diabetes_years: string; hba1c: string };
type Flags = {
    neuropathy: boolean; pad: boolean; prior_ulcer: boolean;
    smoker: boolean; deformity: boolean;
};

// Mirrors main.py: an inline message beats a round trip that ends in a 400.
const RANGES: Record<keyof Numeric, [number, number, string]> = {
    age: [0, 120, "Age"],
    bmi: [8, 90, "BMI"],
    diabetes_years: [0, 90, "Diabetes duration"],
    hba1c: [3, 20, "HbA1c"],
};

// Short labels: the long ones wrapped and knocked the field boxes out of line.
const COMORBIDITIES: { key: keyof Flags; label: string; hint: string }[] = [
    { key: "neuropathy", label: "Neuropathy", hint: "Loss of protective sensation (10 g monofilament)" },
    { key: "pad", label: "PAD", hint: "Peripheral arterial disease — absent pedal pulses or ABI < 0.9" },
    { key: "prior_ulcer", label: "Prior ulcer", hint: "Previous ulcer or amputation — strongest single predictor of recurrence" },
    { key: "deformity", label: "Deformity", hint: "Charcot, hammer toes, prominent metatarsal heads" },
    { key: "smoker", label: "Smoker", hint: "Active smoking — impairs perfusion and wound healing" },
];

const FIELDS: { key: keyof Numeric; label: string; placeholder: string; step: string }[] = [
    { key: "age", label: "Age", placeholder: "55", step: "1" },
    { key: "bmi", label: "BMI", placeholder: "28.5", step: "0.1" },
    { key: "diabetes_years", label: "Diabetes yrs", placeholder: "10", step: "1" },
    { key: "hba1c", label: "HbA1c %", placeholder: "8.2", step: "0.1" },
];

const EMPTY_NUMS: Numeric = { age: "", bmi: "", diabetes_years: "", hba1c: "" };
const EMPTY_FLAGS: Flags = {
    neuropathy: false, pad: false, prior_ulcer: false, smoker: false, deformity: false,
};

export default function UploadForm({
    setResult, setLoading, loading, pending, clearPending,
}: {
    setResult: (r: any) => void;
    setLoading: (b: boolean) => void;
    loading: boolean;
    pending?: { preset: Preset; file: File } | null;
    clearPending?: () => void;
}) {
    const [file, setFile] = useState<File | null>(null);
    const [preview, setPreview] = useState<string | null>(null);
    const [dragging, setDragging] = useState(false);
    const [error, setError] = useState<string | null>(null);
    const [nums, setNums] = useState<Numeric>(EMPTY_NUMS);
    const [flags, setFlags] = useState<Flags>(EMPTY_FLAGS);
    const [presetNote, setPresetNote] = useState<string | null>(null);

    // Object URLs leak if they are never revoked; one preview at a time.
    useEffect(() => {
        if (!file) { setPreview(null); return; }
        const url = URL.createObjectURL(file);
        setPreview(url);
        return () => URL.revokeObjectURL(url);
    }, [file]);

    // Replaced wholesale, so no field survives from a previous preset.
    useEffect(() => {
        if (!pending) return;
        const { preset, file: f } = pending;
        setFile(f);
        setError(null);
        setPresetNote(preset.note);
        const n = { ...EMPTY_NUMS };
        const fl = { ...EMPTY_FLAGS };
        for (const [k, v] of Object.entries(preset.clinical)) {
            if (typeof v === "boolean") fl[k as keyof Flags] = v;
            else n[k as keyof Numeric] = String(v);
        }
        setNums(n);
        setFlags(fl);
        clearPending?.();
    }, [pending, clearPending]);

    const accept = useCallback((f: File | undefined) => {
        if (!f) return;
        if (!f.type.startsWith("image/")) { setError("That file is not an image."); return; }
        if (f.size > 15 * 1024 * 1024) { setError("Image is larger than the 15 MB limit."); return; }
        setError(null);
        setPresetNote(null);
        setFile(f);
    }, []);

    const validate = (): string | null => {
        for (const k of Object.keys(RANGES) as (keyof Numeric)[]) {
            const raw = nums[k];
            if (raw === "") continue;
            const v = Number(raw);
            const [lo, hi, label] = RANGES[k];
            if (Number.isNaN(v)) return `${label} must be a number.`;
            if (v < lo || v > hi) return `${label} must be between ${lo} and ${hi}.`;
        }
        return null;
    };

    const reset = () => {
        setFile(null);
        setNums(EMPTY_NUMS);
        setFlags(EMPTY_FLAGS);
        setPresetNote(null);
        setError(null);
        setResult(null);
    };

    const onSubmit = async (e: React.FormEvent) => {
        e.preventDefault();
        if (!file) { setError("Select an image first, or pick one of the examples below."); return; }
        const bad = validate();
        if (bad) { setError(bad); return; }

        setError(null);
        setLoading(true);
        const data = new FormData();
        data.append("image", file);
        // Blank fields are omitted so the backend keeps them at the reference.
        (Object.keys(nums) as (keyof Numeric)[]).forEach((k) => {
            if (nums[k] !== "") data.append(k, nums[k]);
        });
        (Object.keys(flags) as (keyof Flags)[]).forEach((k) => {
            if (flags[k]) data.append(k, "1");
        });

        try {
            const res = await axios.post(`${API_URL}/predict`, data, { timeout: 120000 });
            setResult({ ...res.data, _previewUrl: preview, _fileName: file.name });
        } catch (err: any) {
            // 400s explain exactly what was wrong; do not flatten them.
            const detail = err?.response?.data?.detail;
            if (detail) setError(typeof detail === "string" ? detail : JSON.stringify(detail));
            else if (err?.code === "ECONNABORTED") setError("The analysis timed out. The model may be waking from a cold start; try again.");
            else if (err?.response) setError(`Server error ${err.response.status}. Please try again.`);
            else setError(`Cannot reach the analysis service at ${API_URL}.`);
            setResult(null);
        } finally {
            setLoading(false);
        }
    };

    const selected = Object.values(flags).filter(Boolean).length;

    return (
        <form onSubmit={onSubmit} noValidate className="panel rounded-2xl p-5 space-y-4">
            <div className="flex items-center justify-between gap-3">
                <h2 className="text-[11px] font-black uppercase tracking-[0.18em] flex items-center gap-2">
                    <span className="sec-num">01</span>
                    <span className="text-muted">Patient input</span>
                </h2>
                {(file || selected > 0) && (
                    <button type="button" onClick={reset}
                        className="text-[10px] font-black uppercase tracking-[0.12em] text-subtle hover:text-danger transition-colors">
                        Reset
                    </button>
                )}
            </div>

            {/* ood_check.py: 100% of whole-foot photos come back as ulcer. */}
            <p className="flex items-start gap-2 px-3 py-2 rounded-lg bg-warn/10 border border-warn/25 text-[11px] text-warn/90 leading-snug">
                <svg className="w-3.5 h-3.5 shrink-0 mt-px" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
                </svg>
                <span><strong>Close-up crop of skin only.</strong> Whole-foot photos are outside the training distribution and come back as ulcer regardless.</span>
            </p>

            <div
                onDragOver={(e) => { e.preventDefault(); setDragging(true); }}
                onDragLeave={() => setDragging(false)}
                onDrop={(e) => { e.preventDefault(); setDragging(false); accept(e.dataTransfer.files?.[0]); }}
                className={`relative rounded-xl border-2 border-dashed transition-colors ${dragging ? "border-brand bg-brand/5" : "border-line/25 hover:border-brand/50 bg-surface2/40"}`}
            >
                <input
                    type="file" accept="image/*"
                    onChange={(e) => accept(e.target.files?.[0])}
                    className="absolute inset-0 w-full h-full opacity-0 cursor-pointer z-20"
                    aria-label="Upload plantar image"
                />
                {file ? (
                    // Beside the filename, not above: stacked doubled the rail height.
                    <div className="flex items-center gap-3 p-2.5 pointer-events-none">
                        <img src={preview ?? ""} alt="Selected scan preview"
                            className="w-14 h-14 rounded-lg object-cover border border-line/20 shrink-0" />
                        <div className="min-w-0">
                            <p className="text-xs font-bold truncate">{file.name}</p>
                            <p className="text-[10px] text-subtle mt-0.5">
                                {(file.size / 1024).toFixed(0)} KB · ready · click to replace
                            </p>
                        </div>
                    </div>
                ) : (
                    <div className="px-4 py-5 text-center pointer-events-none">
                        <svg className="h-6 w-6 text-brand mx-auto mb-2" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-8l-4-4m0 0L8 8m4-4v12" />
                        </svg>
                        <p className="text-[13px] font-bold leading-tight">Drop an image or click</p>
                        <p className="text-[10px] text-subtle mt-1">JPEG, PNG, WebP · max 15 MB</p>
                    </div>
                )}
            </div>

            {presetNote && (
                <p className="text-[11px] text-muted leading-snug panel-inset rounded-lg px-3 py-2">
                    <strong className="text-brand">Example loaded. </strong>{presetNote}
                </p>
            )}

            <div>
                <div className="flex items-baseline justify-between gap-2 mb-2">
                    <p className="text-[10px] font-black uppercase tracking-[0.15em] text-brand">Clinical history</p>
                    <p className="text-[10px] text-subtle">all optional</p>
                </div>
                {/* One-line labels, so the four boxes sit on two clean rows. */}
                <div className="grid grid-cols-2 gap-2.5">
                    {FIELDS.map(({ key, label, placeholder, step }) => (
                        <div key={key}>
                            <label htmlFor={key}
                                className="block text-[10px] font-bold text-subtle uppercase tracking-[0.1em] mb-1 truncate">
                                {label}
                            </label>
                            <input
                                id={key} name={key} type="number" step={step} inputMode="decimal"
                                min={RANGES[key][0]} max={RANGES[key][1]}
                                value={nums[key]}
                                onChange={(e) => setNums({ ...nums, [key]: e.target.value })}
                                placeholder={placeholder}
                                className="field py-2.5 text-sm"
                            />
                        </div>
                    ))}
                </div>
            </div>

            <div>
                <div className="flex items-baseline justify-between gap-2 mb-2">
                    <p className="text-[10px] font-black uppercase tracking-[0.15em] text-brand">Comorbidities</p>
                    <p className="text-[10px] text-subtle tabular-nums">{selected} selected</p>
                </div>
                {/* Toggles, not an accordion: state is visible without opening. */}
                <div className="flex flex-wrap gap-1.5">
                    {COMORBIDITIES.map(({ key, label, hint }) => {
                        const on = flags[key];
                        return (
                            <button
                                key={key} type="button" role="switch" aria-checked={on} title={hint}
                                onClick={() => setFlags({ ...flags, [key]: !on })}
                                className={`px-2.5 py-1.5 rounded-lg border text-[11px] font-bold transition-colors ${on
                                    ? "bg-brand/15 border-brand/40 text-brand"
                                    : "panel-inset text-muted hover:text-fg hover:border-brand/30"}`}
                            >
                                {on && "✓ "}{label}
                            </button>
                        );
                    })}
                </div>
            </div>

            {error && (
                <p role="alert" className="flex items-start gap-2 p-3 rounded-lg bg-danger/10 border border-danger/30 text-danger text-xs leading-snug">
                    <svg className="w-4 h-4 shrink-0 mt-px" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 9v2m0 4h.01M5.07 19h13.86a2 2 0 001.71-3.03l-6.93-12a2 2 0 00-3.42 0l-6.93 12A2 2 0 005.07 19z" />
                    </svg>
                    <span>{error}</span>
                </p>
            )}

            <button type="submit" disabled={loading || !file}
                className="w-full rounded-xl px-5 py-3.5 bg-brand text-brandFg font-black text-xs tracking-[0.15em] uppercase hover:opacity-90 transition-opacity disabled:opacity-40 disabled:cursor-not-allowed flex items-center justify-center gap-2">
                {loading ? (
                    <>
                        <svg className="animate-spin h-4 w-4" fill="none" viewBox="0 0 24 24">
                            <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
                            <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z" />
                        </svg>
                        Analysing…
                    </>
                ) : "Run analysis"}
            </button>
        </form>
    );
}
