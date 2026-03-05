"use client";

export default function ResultCard({ result, loading }: any) {
    if (loading) {
        return (
            <div className="bg-white/[0.02] backdrop-blur-2xl p-8 rounded-[2rem] border border-white/10 shadow-2xl h-full flex flex-col items-center justify-center min-h-[600px] relative overflow-hidden">
                {/* Scanning Animation */}
                <div className="absolute inset-0 bg-[linear-gradient(transparent_0%,rgba(6,182,212,0.1)_50%,transparent_100%)] w-full h-full animate-[scan_2s_ease-in-out_infinite]"></div>
                <div className="relative w-32 h-32 mb-8">
                    <div className="absolute inset-0 border-2 border-dashed border-cyan-500/30 rounded-full animate-[spin_10s_linear_infinite]"></div>
                    <div className="absolute inset-2 border-t-4 border-cyan-400 rounded-full animate-[spin_1.5s_cubic-bezier(0.5,0,0.5,1)_infinite] shadow-[0_0_15px_rgba(34,211,238,0.5)]"></div>
                    <div className="absolute inset-6 border-b-4 border-blue-500 rounded-full animate-[spin_2s_ease-in-out_infinite_reverse]"></div>
                    <div className="absolute inset-0 flex items-center justify-center text-cyan-400">
                        <svg className="w-8 h-8 animate-pulse" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M13 10V3L4 14h7v7l9-11h-7z" /></svg>
                    </div>
                </div>
                <p className="text-cyan-400 font-bold tracking-[0.3em] uppercase text-xs animate-pulse">Extracting Deep Features</p>
            </div>
        );
    }

    if (!result) {
        return (
            <div className="bg-white/[0.02] backdrop-blur-2xl p-8 rounded-[2rem] border border-white/5 border-dashed h-full flex flex-col items-center justify-center min-h-[600px] text-slate-500 group">
                <div className="w-24 h-24 rounded-full bg-white/5 flex items-center justify-center mb-6 group-hover:scale-110 transition-transform duration-500 border border-white/5">
                    <svg className="w-10 h-10 text-slate-600" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1} d="M9 3v2m6-2v2M9 19v2m6-2v2M5 9H3m2 6H3m18-6h-2m2 6h-2M7 19h10a2 2 0 002-2V7a2 2 0 00-2-2H7a2 2 0 00-2 2v10a2 2 0 002 2zM9 9h6v6H9V9z" /></svg>
                </div>
                <p className="text-sm font-medium tracking-widest uppercase">System Awaiting Data</p>
            </div>
        );
    }

    const riskTheme = {
        HIGH: { glow: "shadow-[0_0_50px_rgba(239,68,68,0.2)]", text: "text-red-400", border: "border-red-500/30", bg: "bg-red-500/10", bar: "from-red-600 to-orange-500" },
        MEDIUM: { glow: "shadow-[0_0_50px_rgba(245,158,11,0.2)]", text: "text-amber-400", border: "border-amber-500/30", bg: "bg-amber-500/10", bar: "from-amber-500 to-yellow-400" },
        LOW: { glow: "shadow-[0_0_50px_rgba(16,185,129,0.2)]", text: "text-emerald-400", border: "border-emerald-500/30", bg: "bg-emerald-500/10", bar: "from-emerald-500 to-teal-400" }
    }[result.risk as string] || { glow: "", text: "text-slate-400", border: "border-slate-600", bg: "bg-slate-800", bar: "from-slate-500 to-slate-400" };

    return (
        <div className={`bg-white/[0.03] backdrop-blur-3xl p-8 rounded-[2rem] border ${riskTheme.border} ${riskTheme.glow} h-full flex flex-col relative overflow-hidden transition-all duration-700`}>

            {/* Header */}
            <div className="flex items-center justify-between mb-10 pb-6 border-b border-white/10">
                <div>
                    <h2 className="text-2xl font-black text-white tracking-wide">Diagnostic Matrix</h2>
                    <div className="flex items-center gap-2 mt-2">
                        <span className="w-1.5 h-1.5 rounded-full bg-cyan-500 animate-pulse"></span>
                        <p className="text-cyan-400/80 text-xs font-bold tracking-widest uppercase">Confidence: {(result.confidence * 100).toFixed(1)}%</p>
                    </div>
                </div>
                <div className={`px-6 py-3 rounded-xl font-black tracking-[0.2em] border ${riskTheme.border} ${riskTheme.bg} ${riskTheme.text} flex items-center gap-3 backdrop-blur-md`}>
                    <div className={`w-2 h-2 rounded-full ${riskTheme.bg.replace('/10', '')} animate-ping`}></div>
                    {result.risk} RISK
                </div>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-10 flex-1">

                {/* Visual Proof */}
                <div className="flex flex-col items-center w-full min-w-0">
                    <h3 className="text-[10px] font-black text-slate-400 uppercase tracking-[0.2em] mb-4 w-full text-left">
                        Grad-CAM Spatial Analysis
                    </h3>

                    <div className="relative w-full max-w-[300px] mx-auto aspect-square rounded-2xl overflow-hidden border-2 border-white/10 shadow-2xl bg-black/60 flex items-center justify-center group">

                        <div className="absolute inset-0 bg-[linear-gradient(transparent_0%,rgba(6,182,212,0.1)_50%,transparent_100%)] w-full h-[200%] animate-[scan_3s_linear_infinite] z-10 pointer-events-none"></div>

                        <img
                            src={`data:image/jpeg;base64,${result.heatmap}`}
                            alt="Heatmap"
                            className="absolute inset-0 w-full h-full object-contain z-0 transition-transform duration-700 group-hover:scale-[1.03]"
                        />

                        <div className="absolute bottom-4 left-4 right-4 z-20">
                            <div className="bg-black/80 backdrop-blur-md border border-white/10 rounded-lg p-2 text-center">
                                <p className="text-[9px] text-cyan-300 font-bold uppercase tracking-wider">
                                    Red Zones = Critical Features
                                </p>
                            </div>
                        </div>
                    </div>
                </div>

                {/* SHAP & Justification */}
                <div className="flex flex-col justify-between">

                    <div className="mb-8">
                        <h3 className="text-[10px] font-black text-slate-400 uppercase tracking-[0.2em] mb-6">SHAP Feature Influence</h3>
                        <div className="space-y-6">
                            {result.shap?.map((s: any, i: number) => (
                                <div key={i} className="relative">
                                    <div className="flex justify-between text-xs mb-2">
                                        <span className="text-slate-300 font-bold tracking-wider">{s.feature}</span>
                                        <span className="text-white font-black">{s.value}%</span>
                                    </div>
                                    <div className="h-2 w-full bg-black/50 rounded-full overflow-hidden border border-white/5">
                                        <div
                                            className={`h-full bg-gradient-to-r ${riskTheme.bar} rounded-full transition-all duration-1000 ease-out relative`}
                                            style={{ width: `${s.value}%` }}
                                        >
                                            <div className="absolute inset-0 bg-white/20 w-full animate-[shimmer_2s_infinite]"></div>
                                        </div>
                                    </div>
                                </div>
                            ))}
                        </div>
                    </div>

                    <div className="bg-black/40 backdrop-blur-md p-6 rounded-2xl border border-white/10 relative overflow-hidden group">
                        <div className={`absolute left-0 top-0 bottom-0 w-1 ${riskTheme.bg.replace('/10', '')}`}></div>
                        <h3 className="text-[10px] font-black text-slate-400 uppercase tracking-[0.2em] mb-3">System Justification</h3>
                        <p className="text-slate-300 text-sm leading-relaxed font-medium">{result.explanation}</p>
                    </div>

                </div>
            </div>
        </div>
    );
}