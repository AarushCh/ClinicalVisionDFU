"use client";
import { useEffect, useState } from "react";
import {
    HistoryEntry, clearHistory, listHistory, removeHistory, summarise, toCSV,
} from "@/lib/history";
import { Chip, MiniBars, Tile, toneForRisk } from "./Bento";

const timeAgo = (t: number) => {
    const s = Math.max(0, (Date.now() - t) / 1000);
    if (s < 60) return "just now";
    if (s < 3600) return `${Math.floor(s / 60)}m ago`;
    if (s < 86400) return `${Math.floor(s / 3600)}h ago`;
    return `${Math.floor(s / 86400)}d ago`;
};

export default function HistoryPanel({
    version, synced, onOpen,
}: {
    version: number;              // bumped by the parent after each new analysis
    synced: boolean;              // true when signed in and writing to Postgres
    onOpen?: (e: HistoryEntry) => void;
}) {
    const [entries, setEntries] = useState<HistoryEntry[]>([]);
    const [busy, setBusy] = useState(true);

    useEffect(() => {
        let alive = true;
        listHistory()
            .then((e) => alive && setEntries(e))
            .finally(() => alive && setBusy(false));
        return () => { alive = false; };
    }, [version, synced]);

    const stats = summarise(entries);

    const download = () => {
        const blob = new Blob([toCSV(entries)], { type: "text/csv;charset=utf-8" });
        const url = URL.createObjectURL(blob);
        const a = document.createElement("a");
        a.href = url;
        a.download = `clinicalvision-history-${new Date().toISOString().slice(0, 10)}.csv`;
        a.click();
        URL.revokeObjectURL(url);
    };

    return (
        <div className="space-y-4">
            <div className="flex flex-wrap items-center justify-between gap-3">
                <div>
                    <h2 className="text-lg font-display">Analysis history</h2>
                    <p className="text-[11px] text-subtle mt-0.5">
                        {synced
                            ? "Synced to your account — row-level security means only you can read these."
                            : "Kept in this browser only. Sign in with an account to sync across devices."}
                        {" "}Derived numbers and a 160px thumbnail; never the full image.
                    </p>
                </div>
                <div className="flex items-center gap-2">
                    <Chip tone={synced ? "good" : "muted"}>{synced ? "Synced" : "This device"}</Chip>
                    {entries.length > 0 && (
                        <>
                            <button onClick={download}
                                className="px-3 py-1.5 rounded-lg panel-inset text-[10px] font-black uppercase tracking-[0.12em] text-subtle hover:text-fg transition-colors">
                                Export CSV
                            </button>
                            <button
                                onClick={async () => {
                                    if (confirm(`Delete all ${entries.length} saved analyses? This cannot be undone.`)) {
                                        setEntries(await clearHistory());
                                    }
                                }}
                                className="px-3 py-1.5 rounded-lg panel-inset text-[10px] font-black uppercase tracking-[0.12em] text-subtle hover:text-danger transition-colors">
                                Clear
                            </button>
                        </>
                    )}
                </div>
            </div>

            {busy ? (
                <p className="text-xs text-subtle py-12 text-center">Loading…</p>
            ) : entries.length === 0 ? (
                <div className="panel-inset rounded-2xl py-16 text-center">
                    <p className="text-sm font-bold">No analyses yet</p>
                    <p className="text-xs text-subtle mt-1.5 max-w-sm mx-auto leading-relaxed">
                        Every scan is saved here automatically — risk, probabilities and
                        readings, with a thumbnail. Export the lot as CSV whenever you like.
                    </p>
                </div>
            ) : (
                <>
                    {/* Same shape in every tile, so the four line up. */}
                    <div className="grid grid-cols-2 lg:grid-cols-4 gap-3">
                        <Tile className="flex flex-col justify-between gap-1">
                            <p className="text-[9px] font-black text-subtle uppercase tracking-[0.15em]">Scans</p>
                            <p className="text-2xl font-black leading-none tabular-nums">{stats.n}</p>
                            <p className="text-[10px] text-subtle leading-tight">saved</p>
                        </Tile>
                        <Tile className="flex flex-col justify-between gap-1">
                            <p className="text-[9px] font-black text-subtle uppercase tracking-[0.15em]">Mean fused risk</p>
                            <p className="text-2xl font-black leading-none tabular-nums">{(stats.meanRisk * 100).toFixed(0)}%</p>
                            <p className="text-[10px] text-subtle leading-tight">across all scans</p>
                        </Tile>
                        <Tile className="flex flex-col justify-between gap-1">
                            <p className="text-[9px] font-black text-subtle uppercase tracking-[0.15em]">Breakdown</p>
                            <p className="text-2xl font-black leading-none tabular-nums flex gap-3">
                                <span className="text-danger">{stats.high}</span>
                                <span className="text-warn">{stats.medium}</span>
                                <span className="text-good">{stats.low}</span>
                            </p>
                            <p className="text-[10px] text-subtle leading-tight">high / medium / low</p>
                        </Tile>
                        <Tile className="flex flex-col justify-between gap-1">
                            <p className="text-[9px] font-black text-subtle uppercase tracking-[0.15em]">Risk trend</p>
                            <MiniBars
                                values={entries.slice(0, 24).reverse().map((e) => e.risk_probability)}
                                tones={entries.slice(0, 24).reverse().map((e) => toneForRisk(e.risk))}
                            />
                            <p className="text-[10px] text-subtle leading-tight">oldest → newest</p>
                        </Tile>
                    </div>

                    <div className="grid sm:grid-cols-2 xl:grid-cols-3 2xl:grid-cols-4 gap-3">
                        {entries.map((e) => {
                            const tone = toneForRisk(e.risk);
                            return (
                                <div key={e.id}
                                    className="panel-inset rounded-xl p-3 flex gap-3 hover:border-brand/30 transition-colors group">
                                    {e.thumb ? (
                                        <img src={e.thumb} alt=""
                                            className="w-14 h-14 rounded-lg object-cover border border-line/20 shrink-0" />
                                    ) : (
                                        <div className="w-14 h-14 rounded-lg bg-surface2 border border-line/20 shrink-0" />
                                    )}

                                    <div className="min-w-0 flex-1">
                                        <div className="flex items-center gap-2">
                                            <Chip tone={tone}>{e.risk}</Chip>
                                            <span className="text-[11px] font-bold truncate">{e.fileName}</span>
                                        </div>
                                        <p className="text-[10px] text-subtle mt-1.5 tabular-nums leading-relaxed">
                                            risk {(e.risk_probability * 100).toFixed(0)}% · P(ulcer) {(e.image_probability * 100).toFixed(0)}%
                                            {e.iwgdf_category != null && ` · IWGDF ${e.iwgdf_category}`}
                                            <br />{timeAgo(e.at)}
                                            {e.factors_supplied.length > 0 && ` · ${e.factors_supplied.length} factors`}
                                        </p>
                                    </div>

                                    <div className="flex flex-col gap-1 shrink-0">
                                        {onOpen && (
                                            <button onClick={() => onOpen(e)} title="Load these clinical inputs"
                                                className="p-1.5 rounded-lg text-subtle hover:text-brand transition-colors opacity-0 group-hover:opacity-100 focus:opacity-100">
                                                <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15" />
                                                </svg>
                                            </button>
                                        )}
                                        <button
                                            onClick={async () => setEntries(await removeHistory(e.id))}
                                            title="Delete this entry"
                                            className="p-1.5 rounded-lg text-subtle hover:text-danger transition-colors opacity-0 group-hover:opacity-100 focus:opacity-100">
                                            <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16" />
                                            </svg>
                                        </button>
                                    </div>
                                </div>
                            );
                        })}
                    </div>
                </>
            )}
        </div>
    );
}
