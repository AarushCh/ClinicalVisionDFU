/**
 * Analysis history: localStorage by default, Supabase once signed in.
 *
 * Overlays are ~120 KB of base64 each and localStorage caps near 5 MB, so only a
 * 160px thumbnail is kept. Tables and policies live in supabase/schema.sql.
 */
import { getSignedInClient, supabaseConfigured } from "./supabase";

export type HistoryEntry = {
    id: string;
    at: number;
    fileName: string;
    thumb: string;
    risk: "HIGH" | "MEDIUM" | "LOW" | string;
    risk_probability: number;
    image_probability: number;
    confidence: number;
    predicted_class: string;
    iwgdf_category: number | null;
    attention_area_pct: number | null;
    focality: number | null;
    clinical_logit_shift: number;
    factors_supplied: string[];
    architecture: string;
    inference_ms: number;
};

const KEY = "cv-history";
const MAX_ENTRIES = 60;

const THUMB_PX = 160;

// Re-exported so callers have one import for "is history syncing?".
export { supabaseConfigured };

/** Downscale to a small JPEG data URL. Returns "" if the browser blocks canvas. */
export async function makeThumb(src: string, px = THUMB_PX): Promise<string> {
    try {
        const img = await new Promise<HTMLImageElement>((resolve, reject) => {
            const i = new Image();
            i.onload = () => resolve(i);
            i.onerror = reject;
            i.src = src;
        });
        const scale = Math.min(px / img.width, px / img.height, 1);
        const w = Math.max(1, Math.round(img.width * scale));
        const h = Math.max(1, Math.round(img.height * scale));
        const c = document.createElement("canvas");
        c.width = w;
        c.height = h;
        const ctx = c.getContext("2d");
        if (!ctx) return "";
        ctx.drawImage(img, 0, 0, w, h);
        return c.toDataURL("image/jpeg", 0.7);
    } catch {
        return "";
    }
}

export function toEntry(result: any, fileName: string, thumb: string): HistoryEntry {
    return {
        id: `${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 7)}`,
        at: Date.now(),
        fileName,
        thumb,
        risk: result.risk,
        risk_probability: result.risk_probability,
        image_probability: result.image_probability,
        confidence: result.confidence,
        predicted_class: result.predicted_class,
        iwgdf_category: result.iwgdf?.category ?? null,
        attention_area_pct: result.attention?.attention_area_pct ?? null,
        focality: result.attention?.focality ?? null,
        clinical_logit_shift: result.clinical_logit_shift ?? 0,
        factors_supplied: result.factors_supplied ?? [],
        architecture: result.model?.architecture ?? "",
        inference_ms: result.inference_ms ?? 0,
    };
}

// --- localStorage backend -------------------------------------------------

function readLocal(): HistoryEntry[] {
    try {
        const raw = localStorage.getItem(KEY);
        const parsed = raw ? JSON.parse(raw) : [];
        return Array.isArray(parsed) ? parsed : [];
    } catch {
        return [];
    }
}

function writeLocal(entries: HistoryEntry[]) {
    try {
        localStorage.setItem(KEY, JSON.stringify(entries));
        return true;
    } catch {
        // Quota exceeded: drop the oldest half rather than lose the newest.
        try {
            localStorage.setItem(KEY, JSON.stringify(entries.slice(0, Math.floor(entries.length / 2))));
            return true;
        } catch {
            return false;
        }
    }
}

// --- Supabase backend (lazy) ----------------------------------------------


const ROW_TO_ENTRY = (r: any): HistoryEntry => ({
    id: r.id,
    at: new Date(r.created_at).getTime(),
    fileName: r.file_name ?? "",
    thumb: r.thumb ?? "",
    risk: r.risk,
    risk_probability: r.risk_probability,
    image_probability: r.image_probability,
    confidence: r.confidence,
    predicted_class: r.predicted_class,
    iwgdf_category: r.iwgdf_category,
    attention_area_pct: r.attention_area_pct,
    focality: r.focality,
    clinical_logit_shift: r.clinical_logit_shift ?? 0,
    factors_supplied: r.factors_supplied ?? [],
    architecture: r.architecture ?? "",
    inference_ms: r.inference_ms ?? 0,
});

// --- public API -----------------------------------------------------------

export async function listHistory(): Promise<HistoryEntry[]> {
    const sb = await getSignedInClient();
    if (sb) {
        const { data, error } = await sb
            .from("analyses")
            .select("*")
            .order("created_at", { ascending: false })
            .limit(MAX_ENTRIES);
        if (!error && data) return data.map(ROW_TO_ENTRY);
        // Fall through to local on any error so history never hard-fails.
    }
    return readLocal();
}

export async function addHistory(entry: HistoryEntry): Promise<HistoryEntry[]> {
    const sb = await getSignedInClient();
    if (sb) {
        const { error } = await sb.from("analyses").insert({
            file_name: entry.fileName,
            thumb: entry.thumb,
            risk: entry.risk,
            risk_probability: entry.risk_probability,
            image_probability: entry.image_probability,
            confidence: entry.confidence,
            predicted_class: entry.predicted_class,
            iwgdf_category: entry.iwgdf_category,
            attention_area_pct: entry.attention_area_pct,
            focality: entry.focality,
            clinical_logit_shift: entry.clinical_logit_shift,
            factors_supplied: entry.factors_supplied,
            architecture: entry.architecture,
            inference_ms: entry.inference_ms,
        });
        if (!error) return listHistory();
    }
    const next = [entry, ...readLocal()].slice(0, MAX_ENTRIES);
    writeLocal(next);
    return next;
}

export async function removeHistory(id: string): Promise<HistoryEntry[]> {
    const sb = await getSignedInClient();
    if (sb) {
        const { error } = await sb.from("analyses").delete().eq("id", id);
        if (!error) return listHistory();
    }
    const next = readLocal().filter((e) => e.id !== id);
    writeLocal(next);
    return next;
}

export async function clearHistory(): Promise<HistoryEntry[]> {
    const sb = await getSignedInClient();
    if (sb) {
        // Deletes only this user's rows: RLS scopes the statement to auth.uid().
        const { error } = await sb.from("analyses").delete().neq("id", "");
        if (!error) return [];
    }
    try {
        localStorage.removeItem(KEY);
    } catch { }
    return [];
}

/** Aggregate counts for the summary strip. */
export function summarise(entries: HistoryEntry[]) {
    const n = entries.length;
    const by = { HIGH: 0, MEDIUM: 0, LOW: 0 } as Record<string, number>;
    let risk = 0;
    let ms = 0;
    for (const e of entries) {
        by[e.risk] = (by[e.risk] ?? 0) + 1;
        risk += e.risk_probability ?? 0;
        ms += e.inference_ms ?? 0;
    }
    return {
        n,
        high: by.HIGH ?? 0,
        medium: by.MEDIUM ?? 0,
        low: by.LOW ?? 0,
        meanRisk: n ? risk / n : 0,
        meanMs: n ? ms / n : 0,
        backend: supabaseConfigured ? "Supabase" : "this browser",
    };
}

/** CSV of the whole history, for the export button. */
export function toCSV(entries: HistoryEntry[]): string {
    const cols: (keyof HistoryEntry)[] = [
        "at", "fileName", "risk", "risk_probability", "image_probability",
        "confidence", "predicted_class", "iwgdf_category", "attention_area_pct",
        "focality", "clinical_logit_shift", "architecture", "inference_ms",
    ];
    const esc = (v: any) => {
        const s = String(v ?? "");
        return /[",\n]/.test(s) ? `"${s.replace(/"/g, '""')}"` : s;
    };
    const head = ["timestamp", ...cols.slice(1)].join(",");
    const rows = entries.map((e) =>
        [new Date(e.at).toISOString(), ...cols.slice(1).map((c) => esc(e[c]))].join(",")
    );
    return [head, ...rows].join("\n");
}
