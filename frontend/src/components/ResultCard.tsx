"use client";
import { useRef, useMemo, useState } from "react";
import { useReactToPrint } from "react-to-print";
import PrintableReport from "./PrintableReport";
import type { Session } from "./AuthGate";

const THEME: Record<string, { text: string; border: string; bg: string; dot: string; bar: string }> = {
    HIGH: { text: "text-danger", border: "border-danger/40", bg: "bg-danger/10", dot: "bg-danger", bar: "from-danger to-orange-500" },
    MEDIUM: { text: "text-warn", border: "border-warn/40", bg: "bg-warn/10", dot: "bg-warn", bar: "from-warn to-yellow-400" },
    LOW: { text: "text-good", border: "border-good/40", bg: "bg-good/10", dot: "bg-good", bar: "from-good to-teal-400" },
};
const FALLBACK = { text: "text-muted", border: "border-line/30", bg: "bg-surface2", dot: "bg-subtle", bar: "from-subtle to-muted" };

const pct = (v?: number) => (v == null ? "—" : `${(v * 100).toFixed(1)}%`);

function Stat({ label, value, hint, accent }: any) {
    return (
        <div className="panel-inset rounded-xl p-3">
            <p className="text-[9px] font-black text-subtle uppercase tracking-[0.15em] leading-tight">{label}</p>
            <p className={`text-lg font-black mt-1 leading-none ${accent || ""}`}>{value}</p>
            {hint && <p className="text-[10px] text-subtle leading-tight mt-1">{hint}</p>}
        </div>
    );
}

function Row({ k, v }: { k: string; v: any }) {
    return (
        <div className="flex justify-between gap-4 py-1 border-b border-line/8 last:border-0">
            <span className="text-subtle">{k}</span>
            <span className="font-mono text-right">{String(v)}</span>
        </div>
    );
}

export default function ResultCard({ result, loading, session }: {
    result: any; loading: boolean; session?: Session | null;
}) {
    const reportRef = useRef(null);
    const [view, setView] = useState<"overlay" | "heatmap" | "original">("overlay");

    const reportId = useMemo(
        () => `DFU-${Date.now().toString(36).toUpperCase().slice(-6)}`,
        // eslint-disable-next-line react-hooks/exhaustive-deps
        [result]
    );
    const handlePrint = useReactToPrint({
        contentRef: reportRef,
        documentTitle: `ClinicalVision_${reportId}`,
    });

    if (loading) {
        return (
            <div className="panel rounded-[2rem] p-8 flex flex-col items-center justify-center min-h-[560px] relative overflow-hidden">
                <div className="absolute inset-0 bg-[linear-gradient(transparent_0%,rgb(var(--brand)/0.10)_50%,transparent_100%)] animate-[scan_2s_ease-in-out_infinite]" />
                <div className="relative w-28 h-28 mb-7">
                    <div className="absolute inset-0 border-2 border-dashed border-brand/30 rounded-full animate-[spin_10s_linear_infinite]" />
                    <div className="absolute inset-2 border-t-4 border-brand rounded-full animate-[spin_1.5s_cubic-bezier(0.5,0,0.5,1)_infinite]" />
                    <div className="absolute inset-6 border-b-4 border-blue-500 rounded-full animate-[spin_2s_ease-in-out_infinite_reverse]" />
                </div>
                <p className="text-brand font-bold tracking-[0.3em] uppercase text-xs animate-pulse">Running inference…</p>
            </div>
        );
    }

    if (!result) {
        return (
            <div className="panel rounded-[2rem] p-8 border-dashed flex flex-col items-center justify-center min-h-[560px] text-subtle text-center">
                <div className="w-20 h-20 rounded-full bg-surface/5 flex items-center justify-center mb-5 border border-line/10">
                    <svg className="w-9 h-9" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1} d="M9 3v2m6-2v2M9 19v2m6-2v2M5 9H3m2 6H3m18-6h-2m2 6h-2M7 19h10a2 2 0 002-2V7a2 2 0 00-2-2H7a2 2 0 00-2 2v10a2 2 0 002 2zM9 9h6v6H9V9z" /></svg>
                </div>
                <p className="text-sm font-medium tracking-widest uppercase">Awaiting patient scan</p>
                <p className="text-xs mt-2 max-w-sm">Upload a tissue image, or pick one of the examples to see a full report.</p>
            </div>
        );
    }

    const t = THEME[result.risk] || FALLBACK;
    const att = result.attention || {};
    const iwgdf = result.iwgdf;
    const attribution = result.attribution || [];
    const probs = result.class_probabilities || {};

    const imageSrc =
        view === "original" && result._previewUrl
            ? result._previewUrl
            : `data:image/png;base64,${view === "heatmap" ? result.heatmap_only : (result.overlay || result.heatmap)}`;

    return (
        <div className={`panel rounded-[2rem] p-6 sm:p-7 border ${t.border}`}>
            <div className="flex flex-wrap items-start justify-between gap-4 mb-6 pb-5 border-b border-line/10">
                <div className="min-w-0">
                    <h2 className="text-2xl font-black tracking-tight">Automated Clinical Report</h2>
                    <p className="text-brand text-[11px] font-bold tracking-widest uppercase mt-1.5">
                        {reportId} · {result.model?.architecture} @ {result.model?.input_size}px · {result.inference_ms?.toFixed(0)} ms
                        {session ? ` · ${session.name}` : ""}
                    </p>
                </div>
                <div className="flex items-center gap-3">
                    <button onClick={handlePrint}
                        className="hidden md:flex items-center gap-2 px-4 py-2.5 rounded-xl border border-line/20 bg-surface/5 hover:bg-surface/10 font-bold text-[10px] tracking-widest uppercase transition-colors">
                        <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M17 17h2a2 2 0 002-2v-4a2 2 0 00-2-2H5a2 2 0 00-2 2v4a2 2 0 002 2h2m2 4h6a2 2 0 002-2v-4a2 2 0 00-2-2H9a2 2 0 00-2 2v4a2 2 0 002 2zm8-12V5a2 2 0 00-2-2H9a2 2 0 00-2 2v4h10z" /></svg>
                        Export PDF
                    </button>
                    <div className={`px-5 py-2.5 rounded-xl font-black border ${t.border} ${t.bg} ${t.text} flex items-center gap-2.5`}>
                        <span className="relative flex h-2 w-2">
                            <span className={`animate-ping absolute inline-flex h-full w-full rounded-full ${t.dot} opacity-75`} />
                            <span className={`relative inline-flex rounded-full h-2 w-2 ${t.dot}`} />
                        </span>
                        <span className="tracking-[0.2em] text-sm">{result.risk} RISK</span>
                    </div>
                </div>
            </div>

            {/* Risk, image probability and confidence are three different numbers.
                The original build printed the fused risk under a "Confidence"
                label, so a confidently-healthy foot read as low confidence. */}
            <div className="grid grid-cols-2 lg:grid-cols-4 gap-3 mb-6">
                <Stat label="Fused risk" value={pct(result.risk_probability)} accent={t.text} hint="image + clinical history" />
                <Stat label="P(ulcer) — image" value={pct(result.image_probability)} hint="CNN alone, calibrated" />
                <Stat label="Classifier confidence" value={pct(result.confidence)} hint={`predicts ${String(result.predicted_class || "").replace(/\(.*\)/, "").trim()}`} />
                <Stat label="Clinical shift" value={`${result.clinical_logit_shift >= 0 ? "+" : ""}${result.clinical_logit_shift?.toFixed(2)}`} hint="log-odds from history" />
            </div>

            {/* Image rail + narrative. Only splits at 2xl, so the prose column is
                never squeezed to a few words per line on mid-width screens. */}
            <div className="grid grid-cols-1 2xl:grid-cols-[320px_minmax(0,1fr)] gap-6">
                <div className="space-y-4 min-w-0">
                    <div>
                        <div className="flex gap-1 mb-2 panel-inset p-1 rounded-lg w-fit">
                            {(["overlay", "heatmap", "original"] as const).map((v) => (
                                <button key={v} onClick={() => setView(v)}
                                    disabled={v === "original" && !result._previewUrl}
                                    className={`px-3 py-1.5 rounded-md text-[10px] font-black uppercase tracking-wider transition-colors disabled:opacity-30 ${view === v ? "bg-brand/20 text-brand" : "text-subtle hover:text-fg"}`}>
                                    {v}
                                </button>
                            ))}
                        </div>
                        <div className="relative w-full aspect-square rounded-2xl overflow-hidden border-2 border-line/15 bg-surface2/60">
                            <img src={imageSrc} alt={`${view} view of the analysed scan`}
                                className="absolute inset-0 w-full h-full object-contain" />
                        </div>
                        <p className="text-[10px] text-subtle mt-2 text-center leading-relaxed">
                            {result.cam_mode === "gradcam++" ? "Grad-CAM++" : "Grad-CAM"} attribution for the ulcer class · warm = higher activation
                        </p>
                    </div>

                    <div className="grid grid-cols-2 gap-2">
                        <Stat label="Attention area" value={`${att.attention_area_pct ?? "—"}%`} hint="above 0.5 threshold" />
                        <Stat label="Regions" value={att.attention_regions ?? "—"} hint="discrete hotspots" />
                        <Stat label="Focality" value={att.focality?.toFixed(3) ?? "—"} hint="1.0 = single tight focus" />
                        <Stat label="Peak intensity" value={att.peak_intensity?.toFixed(2) ?? "—"} hint="max activation" />
                    </div>
                </div>

                <div className="space-y-4 min-w-0">
                    <div className="rounded-2xl border border-brand/20 bg-brand/5 p-5 relative overflow-hidden">
                        <div className="absolute left-0 top-0 bottom-0 w-1 bg-brand" />
                        <h4 className="text-[10px] font-black text-brand uppercase tracking-[0.2em] mb-2">
                            Patient summary — {result.report?.headline}
                        </h4>
                        <p className="font-medium text-sm leading-relaxed">{result.report?.patient_summary}</p>
                    </div>

                    {iwgdf && (
                        <div className={`rounded-2xl border ${t.border} ${t.bg} p-4 flex items-center gap-4`}>
                            <div className={`text-3xl font-black ${t.text} w-12 text-center shrink-0`}>{iwgdf.category}</div>
                            <div className="min-w-0">
                                <p className="text-[10px] font-black text-subtle uppercase tracking-[0.2em]">IWGDF 2023 risk category</p>
                                <p className="font-bold text-sm">{iwgdf.label} · screen {iwgdf.screening_interval?.toLowerCase()}</p>
                                <p className="text-muted text-xs mt-0.5">{iwgdf.basis}</p>
                            </div>
                        </div>
                    )}

                    <div className="panel-inset rounded-2xl p-5">
                        <h3 className="text-[10px] font-black text-subtle uppercase tracking-[0.2em]">Feature attribution</h3>
                        <p className="text-[10px] text-subtle mb-4 mt-1">
                            Exact Shapley values of the additive log-odds model · red raises risk, green lowers it
                        </p>
                        <div className="space-y-3">
                            {attribution.slice(0, 8).map((a: any, i: number) => {
                                const up = a.logit_contribution > 0;
                                return (
                                    <div key={i}>
                                        <div className="flex justify-between text-xs mb-1 gap-3">
                                            <span className="font-bold truncate">{a.feature}</span>
                                            <span className="font-black shrink-0 tabular-nums">
                                                {up ? "+" : ""}{a.logit_contribution?.toFixed(2)}
                                                <span className="text-subtle font-medium"> · {a.influence_pct}%</span>
                                            </span>
                                        </div>
                                        <div className="h-1.5 w-full bg-surface2 rounded-full overflow-hidden">
                                            <div className={`h-full rounded-full bg-gradient-to-r ${up ? "from-danger to-orange-500" : "from-good to-teal-400"}`}
                                                style={{ width: `${Math.max(a.influence_pct, 1.5)}%` }} />
                                        </div>
                                    </div>
                                );
                            })}
                        </div>
                    </div>

                    <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
                        <div className="panel-inset rounded-2xl p-5">
                            <h4 className="text-[10px] font-black text-brand uppercase tracking-[0.2em] mb-2 border-b border-line/10 pb-2">Physician assessment</h4>
                            <p className="text-muted text-xs leading-relaxed">{result.report?.clinical_assessment}</p>
                        </div>
                        <div className="panel-inset rounded-2xl p-5">
                            <h4 className="text-[10px] font-black text-brand uppercase tracking-[0.2em] mb-2 border-b border-line/10 pb-2">Grad-CAM pathology</h4>
                            <p className="text-muted text-xs leading-relaxed">{result.report?.visual_analysis}</p>
                        </div>
                    </div>

                    <div className={`p-5 rounded-xl border ${t.border} ${t.bg}`}>
                        <h4 className={`text-[10px] font-black ${t.text} uppercase tracking-[0.2em] mb-3 flex items-center gap-2`}>
                            <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" /></svg>
                            Recommended triage protocol
                        </h4>
                        <ul className="font-medium text-sm space-y-2">
                            {(result.report?.triage_steps || result.report?.triage?.split("\n") || []).map((step: string, i: number) => (
                                <li key={i} className="flex items-start gap-2">
                                    <span className="opacity-50 mt-0.5">•</span>
                                    <span>{step.replace(/^\d+\.\s*/, "")}</span>
                                </li>
                            ))}
                        </ul>
                    </div>

                    {/* Everything the API already returns but the dashboard used to
                        throw away. Collapsed so it informs without dominating. */}
                    <details className="panel-inset rounded-2xl p-5 group">
                        <summary className="cursor-pointer text-[10px] font-black text-subtle uppercase tracking-[0.2em] flex items-center gap-2 select-none">
                            <svg className="w-3.5 h-3.5 transition-transform group-open:rotate-90" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={3} d="M9 5l7 7-7 7" />
                            </svg>
                            Technical details
                        </summary>
                        <div className="grid grid-cols-1 md:grid-cols-2 gap-x-8 gap-y-1 mt-4 text-[11px]">
                            <Row k="Predicted class" v={result.predicted_class} />
                            <Row k="Decision threshold" v="0.50 on P(ulcer)" />
                            {Object.entries(probs).map(([k, v]: any) => (
                                <Row key={k} k={`P(${k.replace(/\(.*\)/, "").trim()})`} v={Number(v).toFixed(4)} />
                            ))}
                            <Row k="Calibration temperature" v={result.model?.temperature?.toFixed(3)} />
                            <Row k="P(ulcer) clamped" v={result.image_probability_clamped ? `yes → ${result.image_probability_used}` : "no"} />
                            <Row k="Clinical log-odds shift" v={result.clinical_logit_shift?.toFixed(4)} />
                            <Row k="Factors supplied" v={(result.factors_supplied || []).length ? result.factors_supplied.join(", ") : "none"} />
                            <Row k="CAM mode" v={result.cam_mode} />
                            <Row k="Peak location (x, y)" v={att.peak_xy ? `${att.peak_xy[0]}, ${att.peak_xy[1]}` : "—"} />
                            <Row k="Mean activation" v={att.mean_intensity} />
                            <Row k="Lesion bbox (norm)" v={att.bbox_xywh_norm ? att.bbox_xywh_norm.join(", ") : "none above threshold"} />
                            <Row k="Architecture" v={`${result.model?.architecture} @ ${result.model?.input_size}px`} />
                            <Row k="Checkpoint format" v={`v${result.model?.checkpoint_format}`} />
                            <Row k="Val macro-F1" v={result.model?.metrics?.val_macro_f1?.toFixed(4) ?? "—"} />
                            <Row k="Inference time" v={`${result.inference_ms} ms`} />
                        </div>
                        {result.model?.metrics?.calibration?.degenerate && (
                            <p className="text-[10px] text-warn mt-4 leading-relaxed">
                                Calibration note: the validation set was perfectly separated, so
                                temperature scaling had no usable signal and fell back to T = 1.0.
                                Confidence values here are raw softmax.
                            </p>
                        )}
                    </details>

                    {result.report?.limitations && (
                        <p className="text-[10px] text-subtle leading-relaxed border-t border-line/10 pt-3">
                            <strong className="text-muted">Limitations. </strong>{result.report.limitations}
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
