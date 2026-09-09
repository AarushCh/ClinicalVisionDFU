"use client";
import { useRef, useMemo, useState } from "react";
import { useReactToPrint } from "react-to-print";
import PrintableReport from "./PrintableReport";
import type { Session } from "@/lib/auth";
import { modelLabel, useModelInfo } from "@/lib/model";
import {
    Chip, Ring, SectionHead, SignedBar, Stat, Tile, TONE, toneForRisk,
} from "./Bento";

function Row({ k, v }: { k: string; v: any }) {
    return (
        <div className="flex justify-between gap-4 py-1 border-b border-line/10 last:border-0">
            <span className="text-subtle">{k}</span>
            <span className="tabular-nums text-right">{String(v)}</span>
        </div>
    );
}

// Shown as buttons rather than printed as prose: the report states the numbers,
// CliniViz explains them on demand.
const QUICK = [
    ["Explain this", "Explain this result in plain English."],
    ["The heatmap", "What is the heatmap actually showing, and where is it looking?"],
    ["Biggest factor", "Which factor influenced the risk score the most, and by how much?"],
    ["How reliable?", "How confident should I be in this result, and what are the limitations?"],
];

export default function ResultCard({ result, loading, session, onAsk }: {
    result: any; loading: boolean; session?: Session | null;
    onAsk?: (question: string) => void;
}) {
    const reportRef = useRef(null);
    const [view, setView] = useState<"overlay" | "heatmap" | "original">("overlay");
    const model = useModelInfo();

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
            <div className="panel neon rounded-3xl p-8 flex flex-col items-center justify-center min-h-[440px] relative overflow-hidden bento-grid-bg">
                <div className="absolute inset-0 bg-[linear-gradient(transparent_0%,rgb(var(--brand)/0.10)_50%,transparent_100%)] animate-[scan_2s_ease-in-out_infinite]" />
                <div className="relative w-28 h-28 mb-7">
                    <div className="absolute inset-0 border-2 border-dashed border-brand/30 rounded-full animate-[spin_10s_linear_infinite]" />
                    <div className="absolute inset-2 border-t-4 border-brand rounded-full animate-[spin_1.5s_cubic-bezier(0.5,0,0.5,1)_infinite]" />
                    <div className="absolute inset-6 border-b-4 border-violet-500 rounded-full animate-[spin_2s_ease-in-out_infinite_reverse]" />
                </div>
                <p className="text-brand font-bold tracking-[0.3em] uppercase text-xs animate-pulse">Running inference…</p>
                <p className="text-[11px] text-subtle mt-2">CNN → calibration → Grad-CAM++ → clinical fusion</p>
            </div>
        );
    }

    if (!result) {
        return (
            <div className="panel rounded-3xl p-7 bento-grid-bg">
                <SectionHead n="01" title="Awaiting patient scan" right={
                    <span className="text-[10px] text-subtle uppercase tracking-widest">Idle</span>
                } />
                <div className="grid gap-6 lg:grid-cols-[minmax(0,1fr)_minmax(0,1.4fr)] items-center">
                    <div>
                        <div className="w-16 h-16 rounded-2xl neon bg-surface2/60 grid place-items-center mb-5">
                            <svg className="w-8 h-8 text-brand" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.2} d="M9 3v2m6-2v2M9 19v2m6-2v2M5 9H3m2 6H3m18-6h-2m2 6h-2M7 19h10a2 2 0 002-2V7a2 2 0 00-2-2H7a2 2 0 00-2 2v10a2 2 0 002 2zM9 9h6v6H9V9z" />
                            </svg>
                        </div>
                        <h3 className="text-2xl font-black text-gradient mb-2 tracking-tight">Upload a scan to begin</h3>
                        <p className="text-sm text-muted leading-relaxed">
                            Or pick one of the examples on the left — no image of your own
                            needed. Every stage below runs on whatever you give it.
                        </p>
                    </div>
                    <div className="grid grid-cols-2 xl:grid-cols-4 gap-3">
                        {[
                            ["Classification", `${modelLabel(model)}, temperature-calibrated, flip TTA`],
                            ["Explanation", "Grad-CAM++ with area, focality and peak readings"],
                            ["Clinical fusion", "Log-odds combination, exact Shapley attribution"],
                            ["Stratification", "IWGDF 2023 category and screening interval"],
                        ].map(([t, d]) => (
                            <Tile key={t} className="flex flex-col gap-1.5">
                                <p className="text-[10px] font-black uppercase tracking-[0.15em] text-brand">{t}</p>
                                <p className="text-[11px] text-subtle leading-snug">{d}</p>
                            </Tile>
                        ))}
                    </div>
                </div>
            </div>
        );
    }

    const tone = toneForRisk(result.risk);
    const t = TONE[tone];
    const att = result.attention || {};
    const iwgdf = result.iwgdf;
    const attribution = result.attribution || [];
    const probs = result.class_probabilities || {};
    const nFactors = (result.factors_supplied || []).length;
    // Colour the IWGDF tile by its own category, not by the fused risk: a
    // category 0 was rendering red whenever the image alone scored high.
    const cat = iwgdf?.category;
    const iwgdfTone = cat == null ? "muted" : cat >= 2 ? "danger" : cat === 1 ? "warn" : "good";

    const imageSrc =
        view === "original" && result._previewUrl
            ? result._previewUrl
            : `data:image/png;base64,${view === "heatmap" ? result.heatmap_only : (result.overlay || result.heatmap)}`;

    return (
        <div className={`panel neon rounded-3xl p-5 sm:p-6 ${t.glow}`}>
            <div className="flex flex-wrap items-center justify-between gap-4 mb-5 pb-4 border-b border-line/10">
                <div className="min-w-0">
                    <h2 className="text-xl font-black tracking-tight text-gradient">Automated Clinical Report</h2>
                    <p className="text-[10px] font-bold tracking-widest uppercase mt-1 text-subtle">
                        <span className="text-brand">{reportId}</span> · {modelLabel(result.model)} · {result.inference_ms?.toFixed(0)} ms
                        {session ? ` · ${session.name}` : ""}
                    </p>
                </div>
                <div className="flex items-center gap-2.5">
                    <button onClick={handlePrint}
                        className="hidden md:flex items-center gap-2 px-3.5 py-2 rounded-xl border border-line/20 bg-surface/5 hover:bg-surface/10 font-bold text-[10px] tracking-widest uppercase transition-colors">
                        <svg className="w-3.5 h-3.5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M17 17h2a2 2 0 002-2v-4a2 2 0 00-2-2H5a2 2 0 00-2 2v4a2 2 0 002 2h2m2 4h6a2 2 0 002-2v-4a2 2 0 00-2-2H9a2 2 0 00-2 2v4a2 2 0 002 2zm8-12V5a2 2 0 00-2-2H9a2 2 0 00-2 2v4h10z" /></svg>
                        Export PDF
                    </button>
                    <div className={`px-4 py-2 rounded-xl font-black border ${t.chip} flex items-center gap-2.5`}>
                        <span className="relative flex h-2 w-2">
                            <span className="animate-ping absolute inline-flex h-full w-full rounded-full opacity-75" style={{ background: t.stroke }} />
                            <span className="relative inline-flex rounded-full h-2 w-2" style={{ background: t.stroke }} />
                        </span>
                        <span className="tracking-[0.2em] text-xs">{result.risk} RISK</span>
                    </div>
                </div>
            </div>

            {/* 01 — risk, P(ulcer) and confidence are three different numbers. */}
            <SectionHead n="01" title="Headline metrics" right={
                <Chip tone={tone}>{result.predicted_class?.replace(/\(.*\)/, "").trim()}</Chip>
            } />
            <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-5 gap-3 mb-6">
                <Tile neon className="grid place-items-center py-4">
                    <Ring value={result.risk_probability} label="Fused risk"
                        caption={nFactors ? `image + ${nFactors} factors` : "no history — equals P(ulcer)"}
                        tone={tone} size={96} />
                </Tile>
                <Tile className="grid place-items-center py-4">
                    <Ring value={result.image_probability} label="P(ulcer)" caption="CNN alone" tone="brand" size={96} />
                </Tile>
                <Tile className="grid place-items-center py-4">
                    <Ring value={result.confidence} label="Confidence" caption="max class probability" tone="brand" size={96} />
                </Tile>
                <Stat label="Clinical shift"
                    value={`${result.clinical_logit_shift >= 0 ? "+" : ""}${result.clinical_logit_shift?.toFixed(2)}`}
                    hint="log-odds from history" />
                <Stat label="IWGDF category" value={iwgdf?.category ?? "—"} hint={iwgdf?.label} tone={iwgdfTone} />
            </div>

            {/* Evidence | narrative | attribution. Two at xl, one below. */}
            <div className="grid grid-cols-1 xl:grid-cols-12 gap-5 items-start">
                {/* A — visual evidence */}
                <div className="space-y-3 min-w-0 xl:col-span-5 2xl:col-span-3">
                    <SectionHead n="02" title="Attribution" right={
                        <div className="flex gap-0.5 panel-inset p-0.5 rounded-lg">
                            {(["overlay", "heatmap", "original"] as const).map((v) => (
                                <button key={v} onClick={() => setView(v)}
                                    disabled={v === "original" && !result._previewUrl}
                                    className={`px-2 py-1 rounded-md text-[9px] font-black uppercase tracking-wider transition-colors disabled:opacity-30 ${view === v ? "bg-brand/20 text-brand" : "text-subtle hover:text-fg"}`}>
                                    {v}
                                </button>
                            ))}
                        </div>
                    } />
                    <div className="relative w-full aspect-square rounded-2xl overflow-hidden neon bg-surface2/60">
                        <img src={imageSrc} alt={`${view} view of the analysed scan`}
                            className="absolute inset-0 w-full h-full object-contain" />
                    </div>
                    <p className="text-[10px] text-subtle text-center leading-relaxed">
                        {result.cam_mode === "gradcam++" ? "Grad-CAM++" : "Grad-CAM"} for the ulcer class · warm = higher activation
                    </p>

                    <div className="grid grid-cols-2 sm:grid-cols-4 xl:grid-cols-2 gap-2.5">
                        <Stat label="Attention area" value={`${att.attention_area_pct ?? "—"}%`} hint="above 0.5" />
                        <Stat label="Regions" value={att.attention_regions ?? "—"} hint="hotspots" />
                        <Stat label="Focality" value={att.focality?.toFixed(3) ?? "—"} hint="1.0 = tight" />
                        <Stat label="Peak" value={att.peak_intensity?.toFixed(2) ?? "—"} hint="max activation" />
                    </div>
                </div>

                {/* B — narrative */}
                <div className="space-y-3 min-w-0 xl:col-span-7 2xl:col-span-5">
                    <div className="rounded-2xl neon bg-brand/5 p-4">
                        <h4 className="text-[10px] font-black text-brand uppercase tracking-[0.2em] mb-1.5">
                            Patient summary — {result.report?.headline}
                        </h4>
                        <p className="font-medium text-sm leading-relaxed">{result.report?.patient_summary}</p>
                    </div>

                    {iwgdf && (
                        <div className={`rounded-2xl border p-4 flex items-center gap-4 ${TONE[iwgdfTone].chip}`}>
                            <div className="text-3xl font-black w-10 text-center shrink-0">{iwgdf.category}</div>
                            <div className="min-w-0">
                                <p className="text-[10px] font-black uppercase tracking-[0.2em] opacity-70">IWGDF 2023 risk category</p>
                                <p className="font-bold text-sm">{iwgdf.label} · screen {iwgdf.screening_interval?.toLowerCase()}</p>
                                <p className="text-xs mt-0.5 opacity-80">{iwgdf.basis}</p>
                            </div>
                        </div>
                    )}

                    {onAsk && (
                        <div className="flex flex-wrap items-center gap-1.5">
                            <span className="text-[10px] font-black uppercase tracking-[0.15em] text-subtle mr-0.5">
                                Ask CliniViz
                            </span>
                            {QUICK.map(([label, q]) => (
                                <button key={label} type="button" onClick={() => onAsk(q)}
                                    className="px-2.5 py-1.5 rounded-lg panel-inset text-[11px] font-bold text-muted hover:text-brand hover:border-brand/40 transition-colors">
                                    {label}
                                </button>
                            ))}
                        </div>
                    )}

                    {/* Full prose is in the PDF export; on screen it is one click away. */}
                    <details className="panel-inset rounded-2xl p-4 group">
                        <summary className="cursor-pointer text-[10px] font-black text-subtle uppercase tracking-[0.2em] flex items-center gap-2 select-none">
                            <svg className="w-3.5 h-3.5 transition-transform group-open:rotate-90" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={3} d="M9 5l7 7-7 7" />
                            </svg>
                            Written assessment
                        </summary>
                        <div className="grid md:grid-cols-2 gap-4 mt-3">
                            <div>
                                <h4 className="text-[10px] font-black text-brand uppercase tracking-[0.2em] mb-1.5">Physician assessment</h4>
                                <p className="text-muted text-xs leading-relaxed">{result.report?.clinical_assessment}</p>
                            </div>
                            <div>
                                <h4 className="text-[10px] font-black text-brand uppercase tracking-[0.2em] mb-1.5">Grad-CAM pathology</h4>
                                <p className="text-muted text-xs leading-relaxed">{result.report?.visual_analysis}</p>
                            </div>
                        </div>
                    </details>

                    <div className={`p-4 rounded-2xl border ${t.chip}`}>
                        <h4 className="text-[10px] font-black uppercase tracking-[0.2em] mb-2.5 flex items-center gap-2">
                            <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" /></svg>
                            Recommended triage protocol
                        </h4>
                        <ul className="font-medium text-[13px] space-y-1.5 text-fg">
                            {(result.report?.triage_steps || result.report?.triage?.split("\n") || []).map((step: string, i: number) => (
                                <li key={i} className="flex items-start gap-2">
                                    <span className="opacity-50 mt-0.5">•</span>
                                    <span>{step.replace(/^\d+\.\s*/, "")}</span>
                                </li>
                            ))}
                        </ul>
                    </div>
                </div>

                {/* C — attribution and provenance */}
                <div className="space-y-3 min-w-0 xl:col-span-12 2xl:col-span-4">
                    <div className="panel-inset rounded-2xl p-4">
                        <SectionHead n="03" title="Feature attribution" />
                        <p className="text-[10px] text-subtle mb-3 -mt-1">
                            Exact Shapley values of the additive log-odds model · red raises risk, green lowers it
                        </p>
                        <div className="grid grid-cols-1 sm:grid-cols-2 2xl:grid-cols-1 gap-x-6 gap-y-2.5">
                            {attribution.slice(0, 8).map((a: any, i: number) => (
                                <div key={i}>
                                    <div className="flex justify-between text-xs mb-1 gap-3">
                                        <span className="font-bold truncate">{a.feature}</span>
                                        <span className="font-black shrink-0 tabular-nums">
                                            {a.logit_contribution > 0 ? "+" : ""}{a.logit_contribution?.toFixed(2)}
                                            <span className="text-subtle font-medium"> · {a.influence_pct}%</span>
                                        </span>
                                    </div>
                                    <SignedBar pct={a.influence_pct} positive={a.logit_contribution > 0} />
                                </div>
                            ))}
                        </div>
                    </div>

                    <details className="panel-inset rounded-2xl p-4 group">
                        <summary className="cursor-pointer text-[10px] font-black text-subtle uppercase tracking-[0.2em] flex items-center gap-2 select-none">
                            <svg className="w-3.5 h-3.5 transition-transform group-open:rotate-90" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={3} d="M9 5l7 7-7 7" />
                            </svg>
                            Technical details
                        </summary>
                        <div className="grid grid-cols-1 md:grid-cols-2 2xl:grid-cols-1 gap-x-8 gap-y-1 mt-3 text-[11px]">
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
                            <p className="text-[10px] text-warn mt-3 leading-relaxed">
                                Calibration note: the validation set was perfectly separated, so
                                temperature scaling had no usable signal and fell back to T = 1.0.
                                Confidence values here are raw softmax.
                            </p>
                        )}
                    </details>

                    {result.report?.limitations && (
                        <details className="panel-inset rounded-2xl p-4 group">
                            <summary className="cursor-pointer text-[10px] font-black text-warn uppercase tracking-[0.2em] flex items-center gap-2 select-none">
                                <svg className="w-3.5 h-3.5 transition-transform group-open:rotate-90" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={3} d="M9 5l7 7-7 7" />
                                </svg>
                                Scope &amp; limitations
                            </summary>
                            <p className="text-[10px] text-subtle leading-relaxed mt-2">{result.report.limitations}</p>
                        </details>
                    )}
                </div>
            </div>

            <div className="hidden">
                <PrintableReport ref={reportRef} result={result} reportId={reportId} />
            </div>
        </div>
    );
}
