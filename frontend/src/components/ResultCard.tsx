"use client";
import { useRef, useMemo, useState } from "react";
import { useReactToPrint } from "react-to-print";
import PrintableReport from "./PrintableReport";

const THEME: Record<string, any> = {
    HIGH: { glow: "shadow-[0_0_50px_rgba(239,68,68,0.2)]", text: "text-red-400", border: "border-red-500/30", bg: "bg-red-500/10", dot: "bg-red-500", bar: "from-red-600 to-orange-500" },
    MEDIUM: { glow: "shadow-[0_0_50px_rgba(245,158,11,0.2)]", text: "text-amber-400", border: "border-amber-500/30", bg: "bg-amber-500/10", dot: "bg-amber-500", bar: "from-amber-500 to-yellow-400" },
    LOW: { glow: "shadow-[0_0_50px_rgba(16,185,129,0.2)]", text: "text-emerald-400", border: "border-emerald-500/30", bg: "bg-emerald-500/10", dot: "bg-emerald-500", bar: "from-emerald-500 to-teal-400" },
};
const FALLBACK = { glow: "", text: "text-slate-400", border: "border-slate-600", bg: "bg-slate-800", dot: "bg-slate-500", bar: "from-slate-500 to-slate-400" };

const pct = (v: number) => `${(v * 100).toFixed(1)}%`;

function Stat({ label, value, hint, accent }: any) {
    return (
        <div className="bg-black/30 rounded-xl p-3 border border-white/5">
            <p className="text-[9px] font-black text-slate-500 uppercase tracking-[0.15em]">{label}</p>
            <p className={`text-lg font-black mt-1 ${accent || "text-white"}`}>{value}</p>
            {hint && <p className="text-[10px] text-slate-500 leading-tight mt-0.5">{hint}</p>}
        </div>
    );
}

export default function ResultCard({ result, loading }: any) {
    const reportRef = useRef(null);
    const [view, setView] = useState<"overlay" | "heatmap" | "original">("overlay");

    const reportId = useMemo(
        () => `DFU-${Date.now().toString(36).toUpperCase().slice(-6)}`,
        [result]
    );
    const handlePrint = useReactToPrint({
        contentRef: reportRef,
        documentTitle: `ClinicalVision_${reportId}`,
    });

    if (loading) {
        return (
            <div className="bg-white/[0.02] backdrop-blur-2xl p-8 rounded-[2rem] border border-white/10 shadow-2xl h-full flex flex-col items-center justify-center min-h-[600px] relative overflow-hidden">
                <div className="absolute inset-0 bg-[linear-gradient(transparent_0%,rgba(6,182,212,0.1)_50%,transparent_100%)] w-full h-full animate-[scan_2s_ease-in-out_infinite]" />
                <div className="relative w-32 h-32 mb-8">
                    <div className="absolute inset-0 border-2 border-dashed border-cyan-500/30 rounded-full animate-[spin_10s_linear_infinite]" />
                    <div className="absolute inset-2 border-t-4 border-cyan-400 rounded-full animate-[spin_1.5s_cubic-bezier(0.5,0,0.5,1)_infinite] shadow-[0_0_15px_rgba(34,211,238,0.5)]" />
                    <div className="absolute inset-6 border-b-4 border-blue-500 rounded-full animate-[spin_2s_ease-in-out_infinite_reverse]" />
                </div>
                <p className="text-cyan-400 font-bold tracking-[0.3em] uppercase text-xs animate-pulse">Running inference…</p>
            </div>
        );
    }

    if (!result) {
        return (
            <div className="bg-white/[0.02] backdrop-blur-2xl p-8 rounded-[2rem] border border-white/5 border-dashed h-full flex flex-col items-center justify-center min-h-[600px] text-slate-500">
                <div className="w-24 h-24 rounded-full bg-white/5 flex items-center justify-center mb-6 border border-white/5">
                    <svg className="w-10 h-10 text-slate-600" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1} d="M9 3v2m6-2v2M9 19v2m6-2v2M5 9H3m2 6H3m18-6h-2m2 6h-2M7 19h10a2 2 0 002-2V7a2 2 0 00-2-2H7a2 2 0 00-2 2v10a2 2 0 002 2zM9 9h6v6H9V9z" /></svg>
                </div>
                <p className="text-sm font-medium tracking-widest uppercase">Awaiting patient scan</p>
            </div>
        );
    }

    const t = THEME[result.risk] || FALLBACK;
    const att = result.attention || {};
    const iwgdf = result.iwgdf;
    const attribution = result.attribution || [];

    const imageSrc =
        view === "original" && result._previewUrl ? result._previewUrl
            : `data:image/png;base64,${view === "heatmap" ? result.heatmap_only : (result.overlay || result.heatmap)}`;

    return (
        <div className={`bg-white/[0.03] backdrop-blur-3xl p-8 rounded-[2rem] border ${t.border} ${t.glow} h-full flex flex-col relative`} id="clinical-report">

            <div className="flex flex-wrap items-center justify-between mb-6 pb-6 border-b border-white/10 gap-4">
                <div>
                    <h2 className="text-2xl font-black text-white tracking-wide">Automated Clinical Report</h2>
                    <p className="text-cyan-400/80 text-xs font-bold tracking-widest uppercase mt-2">
                        {reportId} · {result.model?.architecture ?? "model"} @ {result.model?.input_size ?? "?"}px · {result.inference_ms?.toFixed(0)} ms
                    </p>
                </div>
                <div className="flex items-center gap-4">
                    <button onClick={handlePrint}
                        className="hidden md:flex items-center gap-2 px-4 py-2 bg-white/5 hover:bg-white/10 border border-white/10 rounded-xl text-white font-bold text-xs tracking-widest uppercase transition-all hover:scale-105">
                        <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M17 17h2a2 2 0 002-2v-4a2 2 0 00-2-2H5a2 2 0 00-2 2v4a2 2 0 002 2h2m2 4h6a2 2 0 002-2v-4a2 2 0 00-2-2H9a2 2 0 00-2 2v4a2 2 0 002 2zm8-12V5a2 2 0 00-2-2H9a2 2 0 00-2 2v4h10z" /></svg>
                        Export PDF
                    </button>
                    <div className={`px-6 py-3 rounded-xl font-black border ${t.border} ${t.bg} ${t.text} flex items-center justify-center gap-3 min-w-[180px]`}>
                        <span className="relative flex h-2 w-2 items-center justify-center shrink-0">
                            <span className={`animate-ping absolute inline-flex h-full w-full rounded-full ${t.dot} opacity-75`} />
                            <span className={`relative inline-flex rounded-full h-2 w-2 ${t.dot}`} />
                        </span>
                        <span className="tracking-[0.2em] pl-[0.2em]">{result.risk} RISK</span>
                    </div>
                </div>
            </div>

            {/* Risk and confidence are different quantities and are shown as such.
                The previous build printed the fused risk score under the label
                "Confidence", so a confidently-healthy foot read as low confidence. */}
            <div className="grid grid-cols-2 md:grid-cols-4 gap-3 mb-6">
                <Stat label="Fused risk" value={pct(result.risk_probability)} accent={t.text}
                    hint="image + clinical history" />
                <Stat label="P(ulcer) — image" value={pct(result.image_probability)}
                    hint="CNN alone, calibrated" />
                <Stat label="Classifier confidence" value={pct(result.confidence)}
                    hint={`predicts ${result.predicted_class?.replace(/\(.*\)/, "").trim()}`} />
                <Stat label="Clinical shift"
                    value={`${result.clinical_logit_shift >= 0 ? "+" : ""}${result.clinical_logit_shift?.toFixed(2)}`}
                    hint="log-odds from history" />
            </div>

            <div className="grid grid-cols-1 xl:grid-cols-12 gap-8 flex-1">
                <div className="xl:col-span-5 flex flex-col gap-5">
                    <div>
                        <div className="flex gap-1 mb-2 bg-black/40 p-1 rounded-lg border border-white/5 w-fit">
                            {(["overlay", "heatmap", "original"] as const).map((v) => (
                                <button key={v} onClick={() => setView(v)}
                                    disabled={v === "original" && !result._previewUrl}
                                    className={`px-3 py-1.5 rounded-md text-[10px] font-black uppercase tracking-wider transition-all disabled:opacity-30 ${view === v ? "bg-cyan-500/20 text-cyan-300" : "text-slate-500 hover:text-slate-300"}`}>
                                    {v}
                                </button>
                            ))}
                        </div>
                        <div className="relative w-full aspect-square rounded-2xl overflow-hidden border-2 border-white/10 shadow-2xl bg-black/60 group">
                            <img src={imageSrc} alt={`${view} view of the analysed scan`}
                                className="absolute inset-0 w-full h-full object-contain transition-transform duration-700 group-hover:scale-[1.03]" />
                        </div>
                        <p className="text-[10px] text-slate-500 mt-2 text-center">
                            {result.cam_mode === "gradcam++" ? "Grad-CAM++" : "Grad-CAM"} attribution for the ulcer class · warm = higher activation
                        </p>
                    </div>

                    <div className="grid grid-cols-2 gap-2">
                        <Stat label="Attention area" value={`${att.attention_area_pct ?? "–"}%`} hint="above 0.5 threshold" />
                        <Stat label="Regions" value={att.attention_regions ?? "–"} hint="discrete hotspots" />
                        <Stat label="Focality" value={att.focality?.toFixed(3) ?? "–"} hint="1.0 = single tight focus" />
                        <Stat label="Peak intensity" value={att.peak_intensity?.toFixed(2) ?? "–"} hint="max activation" />
                    </div>
                </div>

                <div className="xl:col-span-7 flex flex-col gap-4">
                    <div className="bg-blue-500/10 backdrop-blur-md rounded-2xl border border-blue-500/20 p-5 relative overflow-hidden">
                        <div className="absolute left-0 top-0 bottom-0 w-1 bg-blue-500" />
                        <h4 className="text-[10px] font-black text-blue-400 uppercase tracking-[0.2em] mb-2">
                            Patient-Friendly Summary — {result.report.headline}
                        </h4>
                        <p className="text-white font-medium text-sm leading-relaxed">{result.report.patient_summary}</p>
                    </div>

                    {iwgdf && (
                        <div className={`rounded-2xl border ${t.border} ${t.bg} p-4 flex items-center gap-4`}>
                            <div className={`text-3xl font-black ${t.text} shrink-0 w-12 text-center`}>{iwgdf.category}</div>
                            <div className="min-w-0">
                                <p className="text-[10px] font-black text-slate-400 uppercase tracking-[0.2em]">IWGDF 2023 Risk Category</p>
                                <p className="text-white font-bold text-sm">{iwgdf.label} · screen {iwgdf.screening_interval.toLowerCase()}</p>
                                <p className="text-slate-400 text-xs mt-0.5">{iwgdf.basis}</p>
                            </div>
                        </div>
                    )}

                    <div className="bg-black/30 rounded-2xl p-5 border border-white/5">
                        <h3 className="text-[10px] font-black text-slate-400 uppercase tracking-[0.2em] mb-1">
                            Feature Attribution
                        </h3>
                        <p className="text-[10px] text-slate-600 mb-4">
                            Exact Shapley values of the additive log-odds model · red raises risk, green lowers it
                        </p>
                        <div className="space-y-3">
                            {attribution.slice(0, 6).map((a: any, i: number) => {
                                const up = a.logit_contribution > 0;
                                return (
                                    <div key={i}>
                                        <div className="flex justify-between text-xs mb-1 gap-2">
                                            <span className="text-slate-300 font-bold truncate">{a.feature}</span>
                                            <span className="text-white font-black shrink-0 tabular-nums">
                                                {up ? "+" : ""}{a.logit_contribution?.toFixed(2)}
                                                <span className="text-slate-500 font-medium"> · {a.influence_pct}%</span>
                                            </span>
                                        </div>
                                        <div className="h-1.5 w-full bg-black/50 rounded-full overflow-hidden">
                                            <div className={`h-full rounded-full ${up ? "bg-gradient-to-r from-red-600 to-orange-500" : "bg-gradient-to-r from-emerald-500 to-teal-400"}`}
                                                style={{ width: `${a.influence_pct}%` }} />
                                        </div>
                                    </div>
                                );
                            })}
                        </div>
                    </div>

                    <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                        <div className="bg-black/40 backdrop-blur-md rounded-2xl border border-white/10 p-5">
                            <h4 className="text-[10px] font-black text-cyan-500 uppercase tracking-[0.2em] mb-2 border-b border-white/10 pb-2">Physician Assessment</h4>
                            <p className="text-slate-300 text-xs leading-relaxed">{result.report.clinical_assessment}</p>
                        </div>
                        <div className="bg-black/40 backdrop-blur-md rounded-2xl border border-white/10 p-5">
                            <h4 className="text-[10px] font-black text-cyan-500 uppercase tracking-[0.2em] mb-2 border-b border-white/10 pb-2">Grad-CAM Pathology</h4>
                            <p className="text-slate-300 text-xs leading-relaxed">{result.report.visual_analysis}</p>
                        </div>
                    </div>

                    <div className={`mt-auto p-5 rounded-xl border ${t.border} ${t.bg}`}>
                        <h4 className={`text-[10px] font-black ${t.text} uppercase tracking-[0.2em] mb-3 flex items-center gap-2`}>
                            <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" /></svg>
                            Recommended Triage Protocol
                        </h4>
                        <ul className="text-white font-medium text-sm space-y-2">
                            {(result.report.triage_steps || result.report.triage.split("\n")).map((step: string, i: number) => (
                                <li key={i} className="flex items-start gap-2">
                                    <span className="opacity-50 mt-0.5">•</span>
                                    <span>{step.replace(/^\d+\.\s*/, "")}</span>
                                </li>
                            ))}
                        </ul>
                    </div>

                    {result.report.limitations && (
                        <p className="text-[10px] text-slate-500 leading-relaxed border-t border-white/5 pt-3">
                            <strong className="text-slate-400">Limitations. </strong>{result.report.limitations}
                        </p>
                    )}
                </div>
            </div>

            <div className="hidden">
                <PrintableReport ref={reportRef} result={result} reportId={reportId} />
            </div>
        </div>
    );
}
