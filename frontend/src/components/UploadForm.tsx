"use client";
import { useState, useRef, useCallback, useEffect } from "react";
import axios from "axios";

const API_URL = process.env.NEXT_PUBLIC_API_URL || "http://127.0.0.1:10000";

type Numeric = { age: string; bmi: string; diabetes_years: string; hba1c: string };
type Flags = {
    neuropathy: boolean; pad: boolean; prior_ulcer: boolean;
    smoker: boolean; deformity: boolean;
};

// Ranges mirror the server-side checks in main.py. Validating here too gives an
// inline message instead of a round trip that ends in a 400.
const RANGES: Record<keyof Numeric, [number, number, string]> = {
    age: [0, 120, "Age"],
    bmi: [8, 90, "BMI"],
    diabetes_years: [0, 90, "Diabetes duration"],
    hba1c: [3, 20, "HbA1c"],
};

const COMORBIDITIES: { key: keyof Flags; label: string; hint: string }[] = [
    { key: "neuropathy", label: "Peripheral neuropathy", hint: "Loss of protective sensation (10 g monofilament)" },
    { key: "pad", label: "Peripheral arterial disease", hint: "Absent pedal pulses or ABI < 0.9" },
    { key: "prior_ulcer", label: "Prior ulcer / amputation", hint: "Strongest single predictor of recurrence" },
    { key: "deformity", label: "Foot deformity", hint: "Charcot, hammer toes, prominent metatarsal heads" },
    { key: "smoker", label: "Active smoker", hint: "Impairs perfusion and wound healing" },
];

export default function UploadForm({ setResult, setLoading, loading }: any) {
    const [file, setFile] = useState<File | null>(null);
    const [preview, setPreview] = useState<string | null>(null);
    const [dragging, setDragging] = useState(false);
    const [error, setError] = useState<string | null>(null);
    const [showAdvanced, setShowAdvanced] = useState(false);
    const [nums, setNums] = useState<Numeric>({ age: "", bmi: "", diabetes_years: "", hba1c: "" });
    const [flags, setFlags] = useState<Flags>({
        neuropathy: false, pad: false, prior_ulcer: false, smoker: false, deformity: false,
    });
    const inputRef = useRef<HTMLInputElement>(null);

    // Object URLs leak if they are never revoked; one preview at a time.
    useEffect(() => {
        if (!file) { setPreview(null); return; }
        const url = URL.createObjectURL(file);
        setPreview(url);
        return () => URL.revokeObjectURL(url);
    }, [file]);

    const accept = useCallback((f: File | undefined) => {
        if (!f) return;
        if (!f.type.startsWith("image/")) { setError("That file is not an image."); return; }
        if (f.size > 15 * 1024 * 1024) { setError("Image is larger than the 15 MB limit."); return; }
        setError(null);
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

    const onSubmit = async (e: React.FormEvent) => {
        e.preventDefault();
        if (!file) { setError("Select a plantar image first."); return; }
        const bad = validate();
        if (bad) { setError(bad); return; }

        setError(null);
        setLoading(true);
        const data = new FormData();
        data.append("image", file);
        // Only send fields the user actually filled in. Empty fields are omitted
        // so the backend leaves them at the population reference rather than
        // treating a blank as a zero.
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
            // The old handler showed "Backend connection failed." for every
            // failure, including 400s that explain exactly what was wrong.
            const detail = err?.response?.data?.detail;
            if (detail) setError(typeof detail === "string" ? detail : JSON.stringify(detail));
            else if (err?.code === "ECONNABORTED") setError("The analysis timed out. The model may be waking from cold start; try again.");
            else if (err?.response) setError(`Server error ${err.response.status}. Please try again.`);
            else setError(`Cannot reach the analysis service at ${API_URL}.`);
            setResult(null);
        } finally {
            setLoading(false);
        }
    };

    const field = (k: keyof Numeric, label: string, placeholder: string, step = "1") => (
        <div className="space-y-2">
            <label htmlFor={k} className="text-[10px] font-black text-cyan-500 uppercase tracking-[0.15em]">
                {label} <span className="text-slate-600 normal-case tracking-normal font-medium">optional</span>
            </label>
            <input
                id={k} name={k} type="number" step={step} inputMode="decimal"
                min={RANGES[k][0]} max={RANGES[k][1]}
                value={nums[k]}
                onChange={(e) => setNums({ ...nums, [k]: e.target.value })}
                placeholder={placeholder}
                className="w-full bg-black/40 border border-white/10 rounded-xl p-4 text-white focus:ring-2 focus:ring-cyan-500/50 focus:border-cyan-500/50 outline-none transition-all placeholder-slate-600 font-medium"
            />
        </div>
    );

    return (
        <div className="bg-white/[0.02] backdrop-blur-2xl p-8 rounded-[2rem] border border-white/10 shadow-[0_8px_32px_0_rgba(0,0,0,0.36)] relative group">
            <div className="absolute inset-0 bg-gradient-to-b from-white/5 to-transparent opacity-0 group-hover:opacity-100 transition-opacity duration-500 rounded-[2rem] pointer-events-none" />

            <h2 className="text-2xl font-bold text-white mb-2 flex items-center gap-3">
                <div className="p-2 bg-cyan-500/20 rounded-lg border border-cyan-500/30 text-cyan-400">
                    <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M16 7a4 4 0 11-8 0 4 4 0 018 0zM12 14a7 7 0 00-7 7h14a7 7 0 00-7-7z" /></svg>
                </div>
                Patient Parameters
            </h2>
            <p className="text-xs text-slate-500 mb-4">
                Only the image is required. Any clinical field left blank sits at the population
                reference and does not move the result.
            </p>

            {/* Measured in backend/ood_check.py: 100% of whole-foot photographs are
                classified as ulcer at genuine-ulcer confidence. The model cannot
                detect that it is out of scope, so the constraint has to be stated
                to the user before they upload rather than inferred afterwards. */}
            <div className="mb-7 flex items-start gap-2.5 px-4 py-3 rounded-xl bg-amber-500/10 border border-amber-500/25">
                <svg className="w-4 h-4 text-amber-400 shrink-0 mt-0.5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
                </svg>
                <p className="text-[11px] text-amber-200/90 leading-relaxed">
                    <strong className="text-amber-300">Upload a close-up crop of the skin, not the whole foot.</strong>{" "}
                    This model was trained only on tightly-cropped tissue patches. Whole-foot
                    photographs fall outside its training distribution and are classified as
                    ulcer almost every time, regardless of what they show.
                </p>
            </div>

            <form onSubmit={onSubmit} className="space-y-6 relative z-10" noValidate>
                <div
                    className="relative cursor-pointer"
                    onDragOver={(e) => { e.preventDefault(); setDragging(true); }}
                    onDragLeave={() => setDragging(false)}
                    onDrop={(e) => { e.preventDefault(); setDragging(false); accept(e.dataTransfer.files?.[0]); }}
                >
                    <div className={`absolute inset-0 rounded-2xl blur-xl transition-all duration-500 ${dragging ? "bg-cyan-400/40" : "bg-cyan-500/20"}`} />
                    <div className={`relative border-2 border-dashed rounded-2xl p-6 text-center bg-black/40 hover:bg-black/60 transition-all duration-300 backdrop-blur-sm overflow-hidden ${dragging ? "border-cyan-400" : "border-white/20 hover:border-cyan-400/50"}`}>
                        <input
                            ref={inputRef} type="file" accept="image/*"
                            onChange={(e) => accept(e.target.files?.[0])}
                            className="absolute inset-0 w-full h-full opacity-0 cursor-pointer z-20"
                            aria-label="Upload plantar image"
                        />
                        <div className="relative z-10">
                            {preview ? (
                                <img src={preview} alt="Selected scan preview"
                                    className="mx-auto mb-3 max-h-40 rounded-xl border border-white/20 object-contain" />
                            ) : (
                                <div className="w-16 h-16 mx-auto bg-white/5 rounded-full flex items-center justify-center mb-4 border border-white/10">
                                    <svg className="h-8 w-8 text-cyan-400" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-8l-4-4m0 0L8 8m4-4v12" /></svg>
                                </div>
                            )}
                            <p className="text-sm font-bold text-white mb-1 break-all px-4">
                                {file ? file.name : "Drop a plantar image, or click to browse"}
                            </p>
                            <p className="text-xs text-slate-400">
                                {file ? `${(file.size / 1024).toFixed(0)} KB · ready` : "JPEG, PNG or WebP · up to 15 MB"}
                            </p>
                        </div>
                    </div>
                </div>

                <div className="grid grid-cols-2 gap-5">
                    {field("age", "Patient Age", "e.g. 55")}
                    {field("bmi", "BMI Index", "e.g. 28.5", "0.1")}
                </div>
                <div className="grid grid-cols-2 gap-5">
                    {field("diabetes_years", "Diabetes Duration", "e.g. 10")}
                    {field("hba1c", "HbA1c (%)", "e.g. 8.2", "0.1")}
                </div>

                <div>
                    <button type="button" onClick={() => setShowAdvanced((s) => !s)}
                        aria-expanded={showAdvanced}
                        className="flex items-center gap-2 text-[10px] font-black text-cyan-500 uppercase tracking-[0.15em] hover:text-cyan-400 transition-colors">
                        <svg className={`w-3.5 h-3.5 transition-transform ${showAdvanced ? "rotate-90" : ""}`}
                            fill="none" stroke="currentColor" viewBox="0 0 24 24">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={3} d="M9 5l7 7-7 7" />
                        </svg>
                        Comorbidities ({Object.values(flags).filter(Boolean).length} selected)
                    </button>

                    {showAdvanced && (
                        <div className="mt-4 space-y-2 bg-black/30 rounded-2xl p-4 border border-white/5">
                            {COMORBIDITIES.map(({ key, label, hint }) => (
                                <label key={key} className="flex items-start gap-3 p-2 rounded-lg hover:bg-white/5 cursor-pointer transition-colors">
                                    <input type="checkbox" checked={flags[key]}
                                        onChange={(e) => setFlags({ ...flags, [key]: e.target.checked })}
                                        className="mt-0.5 w-4 h-4 rounded accent-cyan-500 shrink-0" />
                                    <span className="min-w-0">
                                        <span className="block text-sm font-bold text-white">{label}</span>
                                        <span className="block text-[11px] text-slate-500 leading-snug">{hint}</span>
                                    </span>
                                </label>
                            ))}
                        </div>
                    )}
                </div>

                {error && (
                    <div role="alert" className="flex items-start gap-3 p-4 rounded-xl bg-red-500/10 border border-red-500/30 text-red-300 text-sm">
                        <svg className="w-5 h-5 shrink-0 mt-0.5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 9v2m0 4h.01M5.07 19h13.86a2 2 0 001.71-3.03l-6.93-12a2 2 0 00-3.42 0l-6.93 12A2 2 0 005.07 19z" /></svg>
                        <span>{error}</span>
                    </div>
                )}

                <button type="submit" disabled={loading || !file}
                    className="w-full relative group/btn mt-8 overflow-hidden rounded-xl p-[1px] disabled:opacity-50 disabled:cursor-not-allowed">
                    <span className="absolute inset-0 bg-gradient-to-r from-cyan-500 via-blue-500 to-cyan-500 animate-[spin_3s_linear_infinite] group-hover/btn:opacity-100 opacity-70" />
                    <div className="relative bg-black/80 backdrop-blur-xl px-8 py-5 rounded-xl transition-all group-hover/btn:bg-transparent">
                        <span className="text-white font-bold tracking-wider text-sm flex items-center justify-center gap-2">
                            {loading ? (
                                <>
                                    <svg className="animate-spin h-5 w-5 text-white" fill="none" viewBox="0 0 24 24"><circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" /><path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z" /></svg>
                                    ANALYSING…
                                </>
                            ) : "ENGAGE AI ANALYSIS"}
                        </span>
                    </div>
                </button>
            </form>
        </div>
    );
}
