"use client";
import { useRef, useMemo } from "react";
import { useReactToPrint } from "react-to-print";
import PrintableReport from "./PrintableReport";

export default function ResultCard({ result, loading }: any) {
    const reportRef = useRef(null);

    // Lock the ID so it doesn't spin
    const reportId = useMemo(() => (Math.random() * 100000).toFixed(0), [result]);

    const handlePrint = useReactToPrint({
        contentRef: reportRef,
        documentTitle: `ClinicalVision_DFU_Report`,
    });

    if (loading) {
        return (
            <div className="bg-white/[0.02] backdrop-blur-2xl p-8 rounded-[2rem] border border-white/10 shadow-2xl h-full flex flex-col items-center justify-center min-h-[600px] relative overflow-hidden">
                <div className="absolute inset-0 bg-[linear-gradient(transparent_0%,rgba(6,182,212,0.1)_50%,transparent_100%)] w-full h-full animate-[scan_2s_ease-in-out_infinite]"></div>
                <div className="relative w-32 h-32 mb-8">
                    <div className="absolute inset-0 border-2 border-dashed border-cyan-500/30 rounded-full animate-[spin_10s_linear_infinite]"></div>
                    <div className="absolute inset-2 border-t-4 border-cyan-400 rounded-full animate-[spin_1.5s_cubic-bezier(0.5,0,0.5,1)_infinite] shadow-[0_0_15px_rgba(34,211,238,0.5)]"></div>
                    <div className="absolute inset-6 border-b-4 border-blue-500 rounded-full animate-[spin_2s_ease-in-out_infinite_reverse]"></div>
                </div>
                <p className="text-cyan-400 font-bold tracking-[0.3em] uppercase text-xs animate-pulse">Compiling Medical EMR...</p>
            </div>
        );
    }

    if (!result) {
        return (
            <div className="bg-white/[0.02] backdrop-blur-2xl p-8 rounded-[2rem] border border-white/5 border-dashed h-full flex flex-col items-center justify-center min-h-[600px] text-slate-500">
                <div className="w-24 h-24 rounded-full bg-white/5 flex items-center justify-center mb-6 border border-white/5">
                    <svg className="w-10 h-10 text-slate-600" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1} d="M9 3v2m6-2v2M9 19v2m6-2v2M5 9H3m2 6H3m18-6h-2m2 6h-2M7 19h10a2 2 0 002-2V7a2 2 0 00-2-2H7a2 2 0 00-2 2v10a2 2 0 002 2zM9 9h6v6H9V9z" /></svg>
                </div>
                <p className="text-sm font-medium tracking-widest uppercase">System Awaiting Patient Scan</p>
            </div>
        );
    }

    const riskTheme = {
        HIGH: { glow: "shadow-[0_0_50px_rgba(239,68,68,0.2)]", text: "text-red-400", border: "border-red-500/30", bg: "bg-red-500/10", dot: "bg-red-500", bar: "from-red-600 to-orange-500" },
        MEDIUM: { glow: "shadow-[0_0_50px_rgba(245,158,11,0.2)]", text: "text-amber-400", border: "border-amber-500/30", bg: "bg-amber-500/10", dot: "bg-amber-500", bar: "from-amber-500 to-yellow-400" },
        LOW: { glow: "shadow-[0_0_50px_rgba(16,185,129,0.2)]", text: "text-emerald-400", border: "border-emerald-500/30", bg: "bg-emerald-500/10", dot: "bg-emerald-500", bar: "from-emerald-500 to-teal-400" }
    }[result.risk as string] || { glow: "", text: "text-slate-400", border: "border-slate-600", bg: "bg-slate-800", dot: "bg-slate-500", bar: "from-slate-500 to-slate-400" };

    return (
        <div className={`bg-white/[0.03] backdrop-blur-3xl p-8 rounded-[2rem] border ${riskTheme.border} ${riskTheme.glow} h-full flex flex-col relative`} id="clinical-report">

            {/* Header with Print Button */}
            <div className="flex flex-wrap items-center justify-between mb-8 pb-6 border-b border-white/10 gap-4">
                <div>
                    <h2 className="text-2xl font-black text-white tracking-wide">Automated Clinical Report</h2>
                    <p className="text-cyan-400/80 text-xs font-bold tracking-widest uppercase mt-2">ID: DFU-{reportId} | Confidence: {(result.confidence * 100).toFixed(1)}%</p>
                </div>

                <div className="flex items-center gap-4">
                    <button onClick={handlePrint} className="hidden md:flex items-center gap-2 px-4 py-2 bg-white/5 hover:bg-white/10 border border-white/10 rounded-xl text-white font-bold text-xs tracking-widest uppercase transition-all hover:scale-105">
                        <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M17 17h2a2 2 0 002-2v-4a2 2 0 00-2-2H5a2 2 0 00-2 2v4a2 2 0 002 2h2m2 4h6a2 2 0 002-2v-4a2 2 0 00-2-2H9a2 2 0 00-2 2v4a2 2 0 002 2zm8-12V5a2 2 0 00-2-2H9a2 2 0 00-2 2v4h10z" /></svg>
                        Export PDF
                    </button>

                    <div className={`px-6 py-3 rounded-xl font-black border ${riskTheme.border} ${riskTheme.bg} ${riskTheme.text} flex items-center justify-center gap-3 min-w-[180px]`}>
                        <div className="relative flex h-2 w-2 items-center justify-center shrink-0">
                            <span className={`animate-ping absolute inline-flex h-full w-full rounded-full ${riskTheme.dot} opacity-75`}></span>
                            <span className={`relative inline-flex rounded-full h-2 w-2 ${riskTheme.dot}`}></span>
                        </div>
                        <span className="tracking-[0.2em] pl-[0.2em]">{result.risk} RISK</span>
                    </div>
                </div>
            </div>

            <div className="grid grid-cols-1 xl:grid-cols-12 gap-8 flex-1">

                {/* Left Column: Visuals & SHAP */}
                <div className="xl:col-span-4 flex flex-col gap-6">
                    <div className="relative w-full aspect-square rounded-2xl overflow-hidden border-2 border-white/10 shadow-2xl bg-black/60 group">
                        <img src={`data:image/jpeg;base64,${result.heatmap}`} alt="Heatmap" className="absolute inset-0 w-full h-full object-contain z-0 transition-transform duration-700 group-hover:scale-[1.03]" />
                    </div>

                    <div className="bg-black/30 rounded-2xl p-5 border border-white/5">
                        <h3 className="text-[10px] font-black text-slate-400 uppercase tracking-[0.2em] mb-4">SHAP AI Interpretability</h3>
                        <div className="space-y-4">
                            {result.shap?.map((s: any, i: number) => (
                                <div key={i}>
                                    <div className="flex justify-between text-xs mb-1">
                                        <span className="text-slate-300 font-bold">{s.feature}</span>
                                        <span className="text-white font-black">{s.value}%</span>
                                    </div>
                                    <div className="h-1.5 w-full bg-black/50 rounded-full overflow-hidden">
                                        <div className={`h-full bg-gradient-to-r ${riskTheme.bar} rounded-full`} style={{ width: `${s.value}%` }}></div>
                                    </div>
                                </div>
                            ))}
                        </div>
                    </div>
                </div>

                {/* Right Column: Expanded Medical Text */}
                <div className="xl:col-span-8 flex flex-col gap-4">

                    {/* Patient Summary Card */}
                    <div className="bg-blue-500/10 backdrop-blur-md rounded-2xl border border-blue-500/20 p-5 relative overflow-hidden">
                        <div className="absolute left-0 top-0 bottom-0 w-1 bg-blue-500"></div>
                        <h4 className="text-[10px] font-black text-blue-400 uppercase tracking-[0.2em] mb-2">Patient-Friendly Summary</h4>
                        <p className="text-white font-medium text-sm leading-relaxed">{result.report.patient_summary}</p>
                    </div>

                    <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                        {/* Clinical Assessment */}
                        <div className="bg-black/40 backdrop-blur-md rounded-2xl border border-white/10 p-5">
                            <h4 className="text-[10px] font-black text-cyan-500 uppercase tracking-[0.2em] mb-2 border-b border-white/10 pb-2">Physician Assessment</h4>
                            <p className="text-slate-300 text-sm leading-relaxed">{result.report.clinical_assessment}</p>
                        </div>

                        {/* Visual Analysis */}
                        <div className="bg-black/40 backdrop-blur-md rounded-2xl border border-white/10 p-5">
                            <h4 className="text-[10px] font-black text-cyan-500 uppercase tracking-[0.2em] mb-2 border-b border-white/10 pb-2">Grad-CAM Pathology</h4>
                            <p className="text-slate-300 text-sm leading-relaxed">{result.report.visual_analysis}</p>
                        </div>
                    </div>

                    {/* Triage Action Plan */}
                    <div className={`mt-auto p-5 rounded-xl border ${riskTheme.border} ${riskTheme.bg}`}>
                        <h4 className={`text-[10px] font-black ${riskTheme.text} uppercase tracking-[0.2em] mb-3 flex items-center gap-2`}>
                            <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" /></svg>
                            Recommended Triage Protocol
                        </h4>
                        <ul className="text-white font-medium text-sm space-y-2">
                            {result.report.triage.split('\n').map((step: string, index: number) => (
                                <li key={index} className="flex items-start gap-2">
                                    <span className="opacity-50 mt-0.5">•</span>
                                    <span>{step}</span>
                                </li>
                            ))}
                        </ul>
                    </div>

                </div>
            </div>
            {/* Hidden Printable A4 Report */}
            <div className="hidden">
                <PrintableReport ref={reportRef} result={result} />
            </div>
        </div>
    );
}